"""
Created by Hamid Eghbal-zadeh at 07.07.21
Johannes Kepler University of Linz
"""
import torch
from torch import nn
import torch.nn.init as init
import numpy as np


class VAE(nn.Module):
    """
    conv encoder:
    input size: torch.Size([1, 3, 48, 48])
    output size: torch.Size([1, 64, 15, 15])
    conv_feat_shape = (1, 64, 15, 15)
    conv_feature_dim = 14400

    conv decoder:
    output: torch.Size([1, 32, 22, 22])

    final layer:
    output: torch.Size([1, 3, 48, 48])
    """

    def __init__(self):
        super(VAE, self).__init__()

        latent_dim = 100
        self.conv_feat_shape = (1, 64, 15, 15)

        conv_feature_dim = np.prod(self.conv_feat_shape)

        # decoder:
        self.encoder = nn.Sequential(nn.Conv2d(3, out_channels=32, kernel_size=6, stride=2, padding=0),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(32, out_channels=32, kernel_size=4, stride=1, padding=0),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(32, out_channels=64, kernel_size=3, stride=1, padding=0),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(64, out_channels=64, kernel_size=3, stride=1, padding=0),
                                     nn.LeakyReLU())

        self.fc_mu = nn.Linear(conv_feature_dim, latent_dim)
        self.fc_var = nn.Linear(conv_feature_dim, latent_dim)

        nn.init.kaiming_normal_(self.fc_mu.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.fc_var.weight, mode='fan_out')

        # decoder
        self.decoder_input = nn.Linear(latent_dim, conv_feature_dim)
        nn.init.kaiming_normal_(self.decoder_input.weight, mode='fan_out')

        self.decoder = nn.Sequential(nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=0, output_padding=0),
                                     nn.LeakyReLU(),
                                     nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=0, output_padding=0),
                                     nn.LeakyReLU(),
                                     nn.ConvTranspose2d(32, 32, kernel_size=4, stride=1, padding=0, output_padding=0),
                                     nn.LeakyReLU())

        # final layer

        self.final_layer = nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2, padding=0, output_padding=0)

    def encode(self, input):
        # fwd
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, self.conv_feat_shape[1], self.conv_feat_shape[2], self.conv_feat_shape[3])
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        # numerical stability tricks and all
        logvar = torch.clamp(logvar, min=logvar.min().item() - 1e-7, max=100.)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        sampled_posterior = eps * std + mu
        return sampled_posterior

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        output = self.decode(z)
        return output


def weights_init(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, mode='fan_out')
            init.constant_(m.bias, 0)
        elif type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    init.kaiming_normal_(param.data, mode='fan_out')
                elif 'weight_hh' in name:
                    init.kaiming_normal_(param.data, mode='fan_out')
                elif 'bias' in name:
                    param.data.fill_(0)
