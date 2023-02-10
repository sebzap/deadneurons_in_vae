import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split


def beta_vae_score(vae, ds: torch.utils.data.Dataset, random_state=0):
    """Returns beta-vae-score

    Args:
        vae (Object): VAE model
        ds (torch.utils.data.Dataset): Dataset
        random_state (int, optional): seed. Defaults to 0.

    Returns:
        [[float, float]]: [training_score, test_score]
    """
    x, y = [], []
    dl = DataLoader(ds, batch_size=1000, shuffle=True)
    for (images, labels) in dl:
        y.append(labels.numpy())
        x.append(vae.encode(images).reshape(images.shape[0], -1).detach().cpu().numpy())

    x = np.concatenate(x)
    y = np.concatenate(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random_state)

    model = linear_model.LogisticRegression(random_state=random_state)
    model.fit(x_train, y_train)

    return [model.score(x_train, y_train), model.score(x_test, y_test)]
