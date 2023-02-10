import torch
from torch import nn
from carla_disentanglement.models.beta_vae import BetaVAE, VAEModule


class AnnealedVAE(BetaVAE):
    """
    https://arxiv.org/pdf/1804.03599.pdf
    """

    def __init__(self, model: VAEModule, gamma=1.0, max_c=50, iterations_c=200, reconstruction='bce', tag=None, tag_suffix: str = None):
        self.max_c = max_c
        self.iterations_c = iterations_c
        super().__init__(model, beta=gamma, reconstruction=reconstruction, tag=tag, tag_suffix=tag_suffix)
        self.num_log_values = 6

    def getParamString(self):
        return "_gamma" + str(self.beta) + "_capacity" + str(self.max_c) + "_iterations" + str(self.iterations_c)

    def gauss_vae_regulariser(self, mu: torch.Tensor, log_var: torch.Tensor):
        """
        Compute the regularisation factor for training a VAE
        with a Gaussian latent space.

        with increasing capacity

        Parameters
        ----------
        mu: Tensor
            Mean predictions from variational encoder.
        log_var: Tensor
            log_var predictions from variational encoder.

        Returns
        -------
        kl_div : Tensor
            The Kullback-Leibler divergence between a Gaussian distribution
            with the given parameters and the standard normal distribution.
        """
        capacity = torch.tensor(min(self.max_c, self.max_c * self.global_iter / self.iterations_c), device=self.device)
        kld = super().gauss_vae_regulariser(mu, log_var)
        return (kld - capacity).abs(), kld, capacity

    def calc_loss(self, x_true, batch_size, epoch_data, epoch_step):
        y, (mu, log_var), _ = self.model.forward(x_true)
        kld_loss, kld, c = self.gauss_vae_regulariser(mu, log_var)
        reg_loss = self.beta * kld_loss
        recon_loss = self.reconstruction_loss(y, x_true).div(batch_size)

        loss = recon_loss + reg_loss

        if self.model.use_model_regulariser:
            loss += self.model_regulariser_weight * self.model.model_regulariser()

        epoch_data[epoch_step, 0] = recon_loss.detach()
        epoch_data[epoch_step, 1] = reg_loss.detach()
        epoch_data[epoch_step, 2] = kld.detach()
        epoch_data[epoch_step, 3] = kld_loss.detach()
        epoch_data[epoch_step, 4] = c.detach()

        with torch.no_grad():
            mse = nn.functional.mse_loss(
                self.model.activation(y) if isinstance(self.model.loss_activation, nn.Identity) else y,
                x_true,
                reduction='sum').div(batch_size)
            epoch_data[epoch_step, 5] = mse.detach()

        return loss

    def log_data(self, epoch, epoch_data, type='train'):
        epoch_data = epoch_data.mean(axis=0).cpu().numpy()
        l1, l2 = epoch_data[0], epoch_data[1]

        self.writer.add_scalar('Loss reconstruction/'+type, l1, epoch)
        self.writer.add_scalar('Loss regularisation/'+type, l2, epoch)
        self.writer.add_scalar('KL divergence/'+type, epoch_data[2], epoch)
        self.writer.add_scalar('KL divergence/kld_annealed-'+type, epoch_data[3], epoch)
        self.writer.add_scalar('KL divergence/capacity-'+type, epoch_data[4], epoch)
        self.writer.add_scalar('MSE reconstruction/'+type, epoch_data[5], epoch)
        return l1 + l2
