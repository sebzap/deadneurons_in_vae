from .base import BaseImageEncoder, Flatten3D, Unsqueeze3D, View
from torch import nn
import torch.nn.init as init


class LSConv64S1(BaseImageEncoder):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__(latent_dim, num_channels, image_size)
        assert image_size == 64, 'This model only works with image size 64x64.'

        self.main = nn.Sequential(
            nn.Conv2d(num_channels, 32, 6, 2, 0),          # B,  32, 32, 32
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 4, 1, 0),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 1, 0),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, 1, 0),
            nn.LeakyReLU(),
            Flatten3D(),
            nn.Linear(64*23*23, latent_dim),
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


class GaussianLSConv64S1(LSConv64S1):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__(latent_dim * 2, num_channels, image_size)

        # override value of _latent_dim
        self._latent_dim = latent_dim

    def forward(self, x):
        mu_logvar = self.main(x)
        mu = mu_logvar[:, :self._latent_dim]
        logvar = mu_logvar[:, self._latent_dim:]
        return mu, logvar


class LSConv64DecoderS1(BaseImageEncoder):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__(latent_dim, num_channels, image_size)
        assert image_size == 64, 'This model only works with image size 64x64.'

        self.main = nn.Sequential(
            nn.Linear(latent_dim, 64*23*23),               # B, 256
            nn.LeakyReLU(),
            View((-1, 64, 23, 23)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 64, 3, 1, 0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 3, 1, 0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 32, 4, 1, 0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, num_channels, 6, 2, 0),
        )
        # output shape = bs x 3 x 64 x 64

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
