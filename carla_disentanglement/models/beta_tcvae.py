from typing import Callable
from collections.abc import Sequence
from datetime import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader
from PIL.Image import Image as PILImage
from tqdm import tqdm
from carla_disentanglement.models.beta_vae import BetaVAE, VAEModule
import math


def total_correlation(z, z_mean, z_logvar):
    """Estimate of total correlation on a batch.
    Taken from https://github.com/amir-abdi/disentanglement-pytorch/blob/master/models/betatcvae.py
    Borrowed from https://github.com/google-research/disentanglement_lib/
    Args:
      z: [batch_size, num_latents]-tensor with sampled representation.
      z_mean: [batch_size, num_latents]-tensor with mean of the encoder.
      z_logvar: [batch_size, num_latents]-tensor with log variance of the encoder.
    Returns:
      Total correlation estimated on a batch.
    """
    # Compute log(q(z(x_j)|x_i)) for every sample in the batch, which is a
    # tensor of size [batch_size, batch_size, num_latents]. In the following
    # comments, [batch_size, batch_size, num_latents] are indexed by [j, i, l].
    log_qz_prob = gaussian_log_density(z.unsqueeze(dim=1),
                                       z_mean.unsqueeze(dim=0),
                                       z_logvar.unsqueeze(dim=0))

    # Compute log prod_l p(z(x_j)_l) = sum_l(log(sum_i(q(z(z_j)_l|x_i)))
    # + constant) for each sample in the batch, which is a vector of size
    # [batch_size,].
    log_qz_product = log_qz_prob.exp().sum(dim=1, keepdim=False).log().sum(dim=1, keepdim=False)

    # Compute log(q(z(x_j))) as log(sum_i(q(z(x_j)|x_i))) + constant =
    # log(sum_i(prod_l q(z(x_j)_l|x_i))) + constant.
    log_qz = log_qz_prob.sum(dim=2, keepdim=False).exp().sum(dim=1, keepdim=False).log()

    return (log_qz - log_qz_product).mean()


def gaussian_log_density(samples, mean, log_var):
    """ Estimate the log density of a Gaussian distribution
    Borrowed from https://github.com/google-research/disentanglement_lib/
    :param samples: batched samples of the Gaussian densities with mean=mean and log of variance = log_var
    :param mean: batched means of Gaussian densities
    :param log_var: batches means of log_vars
    :return:
    """
    pi = torch.tensor(math.pi, requires_grad=False)
    normalization = torch.log(2. * pi)
    inv_sigma = torch.exp(-log_var)
    tmp = samples - mean
    return -0.5 * (tmp * tmp * inv_sigma + log_var + normalization)


class BetaTCVAE(BetaVAE):
    """
    https://arxiv.org/pdf/1802.04942.pdf
    """

    def __init__(self, model: VAEModule, beta=1.0, reconstruction='bce', tag=None, tag_suffix: str = None):
        super().__init__(model, beta=beta, reconstruction=reconstruction, tag=tag, tag_suffix=tag_suffix)
        self.num_log_values = 5

    def calc_loss(self, x_true, batch_size, epoch_data, epoch_step):
        y, (mu, log_var), z = self.model.forward(x_true)

        kld = self.gauss_vae_regulariser(mu, log_var)
        t_correlation = total_correlation(z, mu, log_var)

        # if self.global_iter < 1224 * 2:
        #     t_correlation = torch.FloatTensor([0.]).to(self.device)
        # elif self.global_iter % 100 == 0:
        #     print(self.global_iter, t_correlation)

        # if self.global_iter == 610:
        #     for x in epoch_data[595:, :]:
        #         print(x)
        #     raise "foo"

        reg_loss = (self.beta - 1) * t_correlation
        recon_loss = self.reconstruction_loss(y, x_true).div(batch_size)

        loss = recon_loss + kld + reg_loss

        if self.model.use_model_regulariser:
            loss += self.model_regulariser_weight * self.model.model_regulariser()

        epoch_data[epoch_step, 0] = recon_loss.detach()
        epoch_data[epoch_step, 1] = reg_loss.detach()
        epoch_data[epoch_step, 2] = kld.detach()
        epoch_data[epoch_step, 4] = t_correlation.detach()

        # if self.global_iter == 600:
        #     print(epoch_data[epoch_step, :], y[0, 0])
        # if self.global_iter == 601:
        #     print(epoch_data[epoch_step, :], y[0, 0])

        with torch.no_grad():
            mse = nn.functional.mse_loss(
                self.model.activation(y) if isinstance(self.model.loss_activation, nn.Identity) else y,
                x_true,
                reduction='sum').div(batch_size)
            epoch_data[epoch_step, 3] = mse.detach()

        return loss

    def log_data(self, epoch, epoch_data, type='train'):
        epoch_data = epoch_data.detach().mean(axis=0).cpu().numpy()
        l1, l2 = epoch_data[0], epoch_data[1]

        self.writer.add_scalar('Loss reconstruction/'+type, l1, epoch)
        self.writer.add_scalar('Loss regularisation/'+type, l2, epoch)
        self.writer.add_scalar('KL divergence/'+type, epoch_data[2], epoch)
        self.writer.add_scalar('MSE reconstruction/'+type, epoch_data[3], epoch)
        self.writer.add_scalar('Total Correlation/'+type, epoch_data[4], epoch)
        return l1 + l2
