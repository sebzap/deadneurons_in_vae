from typing import Callable
from collections.abc import Sequence
from datetime import datetime
from PIL.Image import Image as PILImage
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from carla_disentanglement.models.beta_vae import BetaVAE, VAEModule


class GecoVAE(BetaVAE):
    """
    https://arxiv.org/pdf/1810.00597.pdf
    https://github.com/denproc/Taming-VAEs/blob/2e44f7ae641082baa54d3ac35742dbcd62fb5e6e/train.py#L25
    """

    def __init__(
            self, model: VAEModule, tolerance=.1, alpha=.99, pretrain=1, lbd_step=100, beta=1.0, lambda_init=1.0, tag=None,
            tag_suffix: str = None):
        self.tol = tolerance
        self.alpha = alpha
        self.pretrain = pretrain
        self.lbd_step = lbd_step
        self.lambda_init = lambda_init

        super().__init__(model, beta=beta, reconstruction='mse', tag=tag, tag_suffix=tag_suffix)

        self.reconstruction_loss = self.calc_reconstruction_loss
        self.num_log_values = 7

    def getParamString(self):
        return "_tolerance" + str(self.tol*255.) + "_255_lambda_init" + str(self.lambda_init) + "_step" + str(self.lbd_step)

    def calc_reconstruction_loss(self, input, target):
        return torch.sum(torch.pow(target - input, 2) - (self.tol**2))

    def calc_loss(self, x_true, batch_size, epoch_data, epoch_step):
        y, (mu, log_var), _ = self.model.forward(x_true)
        kld = self.gauss_vae_regulariser(mu, log_var)
        reg_loss = self.beta * kld

        base_recon_loss = self.reconstruction_loss(y, x_true).div(batch_size)
        recon_loss = self.lambd * base_recon_loss

        loss = recon_loss + reg_loss

        if self.model.use_model_regulariser:
            loss += self.model_regulariser_weight * self.model.model_regulariser()

        epoch_data[epoch_step, 0] = recon_loss.detach()
        epoch_data[epoch_step, 1] = reg_loss.detach()
        epoch_data[epoch_step, 2] = kld.detach()

        epoch_data[epoch_step, 3] = base_recon_loss.detach()
        epoch_data[epoch_step, 4] = self.lambd.detach()

        with torch.no_grad():
            mse = nn.functional.mse_loss(
                self.model.activation(y) if isinstance(self.model.loss_activation, nn.Identity) else y,
                x_true,
                reduction='sum').div(batch_size)
            epoch_data[epoch_step, 6] = mse.detach()

        return loss

    def afterTrainStep(self, epoch_data, epoch_step):
        recon_loss = epoch_data[epoch_step, 3]
        with torch.no_grad():
            if self.global_iter == 1:  # global iter is incremented first thing in training loop
                self.constrain_ma = recon_loss
            else:
                self.constrain_ma = self.alpha * self.constrain_ma.detach() + (1 - self.alpha) * recon_loss
            if self.global_iter % self.lbd_step == 0 and self.train_iter >= self.pretrain:
                self.lambd *= torch.clamp(torch.exp(self.constrain_ma), 0.9, 1.05)

            epoch_data[epoch_step, 5] = self.constrain_ma.detach()

        return super().afterTrainStep(epoch_data, epoch_step)

    def beforeTrain(self):
        self.lambd = torch.FloatTensor([self.lambda_init]).to(self.device)
        return super().beforeTrain()

    def log_data(self, epoch, epoch_data, type='train'):
        epoch_data = epoch_data.mean(axis=0).cpu().numpy()
        l1, l2 = epoch_data[0], epoch_data[1]

        self.writer.add_scalar('Loss reconstruction/'+type, l1, epoch)
        self.writer.add_scalar('Loss regularisation/'+type, l2, epoch)
        self.writer.add_scalar('KL divergence/'+type, epoch_data[2], epoch)

        self.writer.add_scalar('Loss reconstruction/'+type+'-base', epoch_data[3], epoch)
        self.writer.add_scalar('Loss reconstruction/'+type+'-lambda', epoch_data[4], epoch)

        if type == "train":
            self.writer.add_scalar('Loss reconstruction/constrain_ma', epoch_data[5], epoch)

        self.writer.add_scalar('MSE reconstruction/'+type, epoch_data[6], epoch)

        return l1 + l2
