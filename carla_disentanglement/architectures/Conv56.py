from .base import BaseImageEncoder, Flatten3D, Unsqueeze3D, View
from torch import nn


class Conv56(BaseImageEncoder):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__(latent_dim, num_channels, image_size)
        assert image_size == 56, 'This model only works with image size 56x56.'

        self.main = nn.Sequential(
            nn.Conv2d(num_channels, 32, 4, 2, 1),  # B,  32, 28, 28
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 14, 14
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  7,  7
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 2, 1),          # B,  32,  4,  4
            nn.ReLU(True),
            Flatten3D(),                         # B, 512
            nn.Linear(32*4*4, 256),              # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, latent_dim),          # B, z_dim
        )

        self.init_layers()

    def forward(self, x):
        return self.main(x)


class GaussianConv56(Conv56):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__(latent_dim * 2, num_channels, image_size)

        # override value of _latent_dim
        self._latent_dim = latent_dim

    def forward(self, x):
        mu_logvar = self.main(x)
        mu = mu_logvar[:, :self._latent_dim]
        logvar = mu_logvar[:, self._latent_dim:]
        return mu, logvar


class Conv56Decoder(BaseImageEncoder):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__(latent_dim, num_channels, image_size)
        assert image_size == 56, 'This model only works with image size 56x56.'

        self.main = nn.Sequential(
            nn.Linear(latent_dim, 256),               # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, 32*4*4),              # B, 512
            nn.ReLU(True),
            View((-1, 32, 4, 4)),                 # B,  32,  4,  4
            nn.ConvTranspose2d(32, 32, 3, 2, 1),  # B,  32,  7,  7
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 14,  14
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 28, 28
            nn.ReLU(True),
            nn.ConvTranspose2d(32, num_channels, 4, 2, 1),  # B,  nc, 56, 56
        )
        # output shape = bs x 3 x 64 x 64

        self.init_layers()

    def forward(self, x):
        return self.main(x)
