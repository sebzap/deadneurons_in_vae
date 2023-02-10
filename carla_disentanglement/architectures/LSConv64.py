from .base import BaseImageEncoder, Flatten3D, Unsqueeze3D, View
from torch import nn


class LSConv64(BaseImageEncoder):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__(latent_dim, num_channels, image_size)
        assert image_size == 64, 'This model only works with image size 64x64.'

        self.main = nn.Sequential(
            nn.Conv2d(num_channels, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  32,  8,  8
            nn.LeakyReLU(),
            Flatten3D(),                         # B, 512
            nn.Linear(64*8*8, latent_dim),              # B, 256
        )

        self.init_layers()

    def forward(self, x):
        return self.main(x)


class GaussianLSConv64(LSConv64):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__(latent_dim * 2, num_channels, image_size)

        # override value of _latent_dim
        self._latent_dim = latent_dim

    def forward(self, x):
        mu_logvar = self.main(x)
        mu = mu_logvar[:, :self._latent_dim]
        logvar = mu_logvar[:, self._latent_dim:]
        return mu, logvar


class LSConv64Decoder(BaseImageEncoder):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__(latent_dim, num_channels, image_size)
        assert image_size == 64, 'This model only works with image size 64x64.'

        self.main = nn.Sequential(
            nn.Linear(latent_dim, 64*8*8),               # B, 256
            nn.LeakyReLU(),
            View((-1, 64, 8, 8)),                 # B,  32,  4,  4
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, num_channels, 4, 2, 1),  # B,  nc, 64, 64
        )
        # output shape = bs x 3 x 64 x 64

        self.init_layers()

    def forward(self, x):
        return self.main(x)
