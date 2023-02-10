from .base import BaseImageEncoder, Flatten3D, Unsqueeze3D, View
from torch import nn
import numpy as np
import torch.nn.init as init


class LeakyConv64(BaseImageEncoder):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__(latent_dim, num_channels, image_size)
        assert image_size == 64, 'This model only works with image size 64x64.'

        self.conv_feat_shape = (1, 64, 24, 24)

        conv_feature_dim = np.prod(self.conv_feat_shape)

        self.main = nn.Sequential(nn.Conv2d(num_channels, out_channels=32, kernel_size=4, stride=2, padding=0),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(32, out_channels=32, kernel_size=4, stride=1, padding=0),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(32, out_channels=64, kernel_size=3, stride=1, padding=0),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(64, out_channels=64, kernel_size=3, stride=1, padding=0),
                                  nn.LeakyReLU(),
                                  Flatten3D(),
                                  nn.Linear(conv_feature_dim, latent_dim))

        self.init_layers()

    def forward(self, x):
        return self.main(x)

    def init_layers(net):
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


class GaussianLeakyConv64(LeakyConv64):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__(latent_dim * 2, num_channels, image_size)

        # override value of _latent_dim
        self._latent_dim = latent_dim

    def forward(self, x):
        mu_logvar = self.main(x)
        mu = mu_logvar[:, :self._latent_dim]
        logvar = mu_logvar[:, self._latent_dim:]
        return mu, logvar


class LeakyConv64Decoder(BaseImageEncoder):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__(latent_dim, num_channels, image_size)
        assert image_size == 64, 'This model only works with image size 64x64.'

        self.conv_feat_shape = (1, 64, 24, 24)

        conv_feature_dim = np.prod(self.conv_feat_shape)

        self.main = nn.Sequential(
            nn.Linear(latent_dim, conv_feature_dim),
            View((-1, 64, 24, 24)),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=0, output_padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=0, output_padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=1, padding=0, output_padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, num_channels, kernel_size=4, stride=2, padding=0, output_padding=0)
        )

        self.init_layers()

    def forward(self, x):
        return self.main(x)

    def init_layers(net):
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
