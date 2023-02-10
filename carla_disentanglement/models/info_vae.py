from typing import Callable
from collections.abc import Sequence
from datetime import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader
from PIL.Image import Image as PILImage
from tqdm import tqdm
from carla_disentanglement.models.beta_vae import BetaVAE


def compute_kernel(x, y):
    x_size = x.shape[0]
    y_size = y.shape[0]
    dim = x.shape[1]
    tiled_x = x.view([x_size, 1, dim]).float().repeat([1, y_size, 1])
    tiled_y = y.view([1, y_size, dim]).float().repeat([x_size, 1, 1])
    return torch.exp(-torch.mean((tiled_x - tiled_y).float() ** 2, dim=2) / dim)


class InfoVAE(BetaVAE):
    """
    https://arxiv.org/pdf/1706.02262.pdf
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module, alpha=0.0, lambd=1.0, mmd_samples=1000, reconstruction='bce'):
        super().__init__(encoder, decoder, beta=(1-alpha), reconstruction=reconstruction)
        self.alpha = alpha
        self.lambd = lambd
        self.mmd_weight = alpha + lambd - 1
        self.mmd_samples = mmd_samples

    def compute_mmd(self, x, y):
        """https://github.com/amir-abdi/disentanglement-pytorch/blob/master/models/infovae.py

        Args:
            x (torch.Tensor): input 1
            y (torch.Tensor): input 2

        Returns:
            torch.Tensor: Maximum Mean Discrepancy
        """
        x_kernel = compute_kernel(x, x)
        y_kernel = compute_kernel(y, y)
        xy_kernel = compute_kernel(x, y)
        return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)

    def train(self, ds: torch.utils.data.Dataset,
              batch_size=256,
              epochs=100,
              loss_every=1,
              loss_callback: Callable[[Sequence], None] = None,
              eval_every=10,
              eval_callback: Callable[[Sequence, Sequence], None] = None,
              reconstruction_every=10,
              reconstruction_callback: Callable[[PILImage], None] = None,
              save_every=20,
              save_path="./checkpoints/"):
        """Start training

        Args:
            ds (torch.utils.data.Dataset): Dataset
            batch_size (int, optional): batch size. Defaults to 256.
            epochs (int, optional): number of epoches. Defaults to 100.
            loss_every (int, optional): after how many epochs loss_callback will be called. Defaults to 1.
            loss_callback (Callable[[Sequence], None], optional): Callback function with [complete loss, reconstruction loss, kl loss]. Defaults to None.
            eval_every (int, optional): after how many epochs evaluation will be perforemd. Defaults to 10.
            eval_callback (Callable[[Sequence, Sequence], None], optional): Callback function with [complete loss, reconstruction loss, kl loss]. Defaults to None.
            reconstruction_every (int, optional): after how many epochs reconstruction_callback will be called. Defaults to 10.
            reconstruction_callback (Callable[[PILImage], None], optional): callback with reconstructed image. Defaults to None.
            save_every (int, optional): after how many epochs model will be saved. Defaults to 20.
            save_path (str, optional): path for model state dict. Defaults to "./checkpoints/".
        """

        train_size = int(0.75 * len(ds))
        test_size = len(ds) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(ds, [train_size, test_size], generator=torch.Generator().manual_seed(0))

        data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
        self.model.train()

        last_save = None

        with tqdm(range(epochs)) as bar:
            for epoch in bar:
                self.train_iter = epoch
                epoch_loss = torch.empty((len(data_loader), 3), dtype=torch.float32, device=self.device)
                for i, (x_true, _) in enumerate(data_loader):
                    x_true = x_true.to(self.device)

                    y, (mu, log_var), z = self.model.forward(x_true)
                    reg_loss = self.beta * self.gauss_vae_regulariser(mu, log_var)
                    recon_loss = self.reconstruction_loss(y, x_true).div(batch_size)
                    mmd_loss = self.mmd_weight * self.compute_mmd(torch.randn(self.mmd_samples, self.z_dim, device=self.device), z)

                    loss = recon_loss + reg_loss + mmd_loss

                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward(retain_graph=False)
                    self.optimizer.step()

                    epoch_loss[i, 0] = recon_loss.detach()
                    epoch_loss[i, 1] = reg_loss.detach()
                    epoch_loss[i, 2] = mmd_loss.detach()

                epoch_loss = epoch_loss.mean(axis=0).cpu().numpy()
                l1, l2, l3 = epoch_loss[0], epoch_loss[1], epoch_loss[2]

                self.losses.append([l1+l2+l3, l1, l2, l3])

                if epoch % save_every == 0:
                    last_save = "vae_" + str(epoch) + '_' + \
                        datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_"+str(l1+l2)+".save"
                    torch.save(self.model.state_dict(), save_path+last_save)

                bar.set_postfix(epoch_stat=epoch, reconstruction_loss=l1, kl_loss=l2, mmd_loss=l3, last_save=last_save)

                if loss_callback:
                    if epoch % loss_every == 0:
                        loss_callback(self.losses)

                if eval_callback:
                    if epoch % eval_every == 0:
                        eval_loss, eval_scores = self.eval(test_dataset, batch_size=batch_size)
                        self.model.train()
                        self.eval_losses.append(eval_loss)
                        self.scores.append(eval_scores)
                        eval_callback(self.eval_losses, self.scores)

                if reconstruction_callback:
                    if epoch % reconstruction_every == 0:
                        reconstruction_callback(self.visualize_reconstruction(ds))

        if loss_callback:
            loss_callback(self.losses)

        if reconstruction_callback:
            reconstruction_callback(self.visualize_reconstruction(ds))

        last_save = "vae_" + \
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_"+str(l1+l2)+".save"
        torch.save(self.model.state_dict(), save_path+last_save)
        print("saved "+last_save)

    def eval(self, ds, batch_size=256):
        self.model.eval()
        data_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        with torch.no_grad():
            epoch_loss = torch.empty((len(data_loader), 3), dtype=torch.float32, device=self.device)
            for i, (x_true, _) in enumerate(data_loader):
                x_true = x_true.to(self.device)

                y, (mu, log_var) = self.model.eval_forward(x_true)
                reg_loss = self.beta * self.gauss_vae_regulariser(mu, log_var)
                recon_loss = self.reconstruction_loss(y, x_true).div(batch_size)
                mmd_loss = self.mmd_weight * self.compute_mmd(torch.randn(self.mmd_samples, self.z_dim, device=self.device), mu)

                epoch_loss[i, 0] = recon_loss.detach()
                epoch_loss[i, 1] = reg_loss.detach()
                epoch_loss[i, 2] = mmd_loss.detach()

            epoch_loss = epoch_loss.mean(axis=0).cpu().numpy()
            l1, l2, l3 = epoch_loss[0], epoch_loss[1], epoch_loss[2]

            return [l1+l2+l3, l1, l2, l3], self.score(ds)
