from .base import BaseImageEncoder, Flatten3D, Unsqueeze3D, View
from torch import nn


class Conv64(BaseImageEncoder):
    def __init__(self, latent_dim, num_channels, image_size, activation_layer=nn.ReLU):
        super().__init__(latent_dim, num_channels, image_size)
        assert image_size == 64, 'This model only works with image size 64x64.'

        self.main = nn.Sequential(
            nn.Conv2d(num_channels, 32//5*4, 4, 2, 1),          # B,  32, 32, 32
            activation_layer(),
            nn.Conv2d(32//5*4, 32//3*2, 4, 2, 1),          # B,  32, 16, 16
            activation_layer(),
            nn.Conv2d(32//3*2, 64//3*2, 4, 2, 1),          # B,  32,  8,  8
            activation_layer(),
            nn.Conv2d(64//3*2, 64//5*4, 4, 2, 1),          # B,  32,  4,  4
            activation_layer(),
            Flatten3D(),                         # B, 1024
            nn.Linear(64//5*4*4*4, 256//2),              # B, 256
            activation_layer(),
            nn.Linear(256//2, latent_dim),          # B, z_dim
        )

        self.init_layers()

    def forward(self, x):
        return self.main(x)


class GaussianConv64(Conv64):
    def __init__(self, latent_dim, num_channels, image_size, activation_layer=nn.ReLU):
        super().__init__(latent_dim * 2, num_channels, image_size, activation_layer=activation_layer)

        # override value of _latent_dim
        self._latent_dim = latent_dim

    def forward(self, x):
        mu_logvar = self.main(x)
        mu = mu_logvar[:, :self._latent_dim]
        logvar = mu_logvar[:, self._latent_dim:]
        return mu, logvar


class Conv64Decoder(BaseImageEncoder):
    def __init__(self, latent_dim, num_channels, image_size, activation_layer=nn.ReLU):
        super().__init__(latent_dim, num_channels, image_size)
        assert image_size == 64, 'This model only works with image size 64x64.'

        self.main = nn.Sequential(
            nn.Linear(latent_dim, 256),               # B, 256
            activation_layer(),
            nn.Linear(256, 64*4*4),              # B, 1024
            nn.ReLU(True),
            View((-1, 64, 4, 4)),                 # B,  32,  4,  4
            nn.ConvTranspose2d(64, 64, 4, 2, 1),  # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, num_channels, 4, 2, 1),  # B,  nc, 64, 64
        )
        # output shape = bs x 3 x 64 x 64

        self.init_layers()

    def forward(self, x):
        return self.main(x)
