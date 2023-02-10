import numpy as np
import torch
from torch.utils.data import Dataset
from disentanglement_lib.data.ground_truth.util import SplitDiscreteStateSpace


class DSpritesDataset(Dataset):
    """
    https://github.com/deepmind/dsprites-dataset

    dataset is labeld whith true shape class
    """

    def __init__(self):
        data = np.load('./carla_disentanglement/data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz',
                       encoding='bytes', allow_pickle=True)
        self.data_tensor = torch.from_numpy(data['imgs']).unsqueeze(1).float()
        self.data_shape = [1, 64, 64]
        # first label is color which is always white
        # and only one class breaks some metrics
        # so first label is cut from data
        self.labels = torch.from_numpy(data['latents_values'][:, 1:])
        self.factor_sizes = np.array(data["metadata"][()][b'latents_sizes'][1:], dtype=np.int64)
        self.latent_factor_indices = list(range(5))
        self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(self.factor_sizes)
        self.state_space = SplitDiscreteStateSpace(self.factor_sizes,
                                                   self.latent_factor_indices)

    def __getitem__(self, index):
        return self.data_tensor[index], self.labels[index]

    def __len__(self):
        return self.data_tensor.size(0)

    @property
    def num_factors(self):
        return self.state_space.num_latent_factors

    @property
    def factors_num_values(self):
        return self.factor_sizes

    @property
    def observation_shape(self):
        return self.data_shape

    def sample(self, num, random_state):
        """Sample a batch of factors Y and observations X."""
        factors = self.sample_factors(num, random_state)
        return factors, self.sample_observations_from_factors(factors, random_state)

    def sample_observations(self, num, random_state):
        """Sample a batch of observations X."""
        return self.sample(num, random_state)[1]

    def sample_factors(self, num, random_state):
        """Sample a batch of factors Y."""
        return self.state_space.sample_latent_factors(num, random_state)

    def sample_observations_from_factors(self, factors, random_state):
        return self.sample_observations_from_factors_no_color(factors, random_state)

    def sample_observations_from_factors_no_color(self, factors, random_state):
        """Sample a batch of observations X given a batch of factors Y."""
        all_factors = self.state_space.sample_all_factors(factors, random_state)
        indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)
        return self.data_tensor[indices]

    def _sample_factor(self, i, num, random_state):
        return random_state.randint(self.factor_sizes[i], size=num)


class DSpritesDatasetMini(Dataset):
    """
    https://github.com/deepmind/dsprites-dataset

    dataset is labeld whith true shape class
    smaler dataset from https://github.com/amir-abdi/disentanglement-pytorch
    """

    def __init__(self):
        data = np.load('./data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64_mini.npz', encoding='bytes')
        self.data_tensor = torch.from_numpy(data['imgs']).unsqueeze(1).float()
        self.labels = torch.from_numpy(data['latents_values'][:, 1])

    def __getitem__(self, index):
        return self.data_tensor[index], self.labels[index]

    def __len__(self):
        return self.data_tensor.size(0)
