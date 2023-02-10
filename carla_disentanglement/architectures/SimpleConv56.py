from .base import BaseImageEncoder, Flatten3D, Unsqueeze3D
from torch import nn


class SimpleConv56(BaseImageEncoder):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__(latent_dim, num_channels, image_size)
        assert image_size == 56, 'This model only works with image size 56x56.'

        self.main = nn.Sequential(
            nn.Conv2d(num_channels, 32, 4, 2, 1),  # 28x28
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # 14x14
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # 7x7
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 1, 0),  # 4x4
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),  # 2x2
            nn.ReLU(True),
            nn.Conv2d(256, 256, 4, 2, 1),  # 2x2
            nn.ReLU(True),
            Flatten3D(),
            nn.Linear(256, latent_dim, bias=True)
        )

        self.init_layers()

    def forward(self, x):
        return self.main(x)


class SimpleGaussianConv56(SimpleConv56):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__(latent_dim * 2, num_channels, image_size)

        # override value of _latent_dim
        self._latent_dim = latent_dim

    def forward(self, x):
        mu_logvar = self.main(x)
        mu = mu_logvar[:, :self._latent_dim]
        logvar = mu_logvar[:, self._latent_dim:]
        return mu, logvar


class SimpleConv56Decoder(BaseImageEncoder):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__(latent_dim, num_channels, image_size)
        assert image_size == 56, 'This model only works with image size 56x56.'

        self.main = nn.Sequential(
            nn.Linear(latent_dim, 256),
            Unsqueeze3D(),
            # nn.Conv2d(latent_dim, 256, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),  # 2x2
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2),  # 6x6
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),  # 12x12
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2),  # 26x26
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2),  # 54x54
            nn.ReLU(True),
            nn.ConvTranspose2d(64, num_channels, 3, 1)  # 56x56
        )

        self.init_layers()

    def forward(self, x):
        return self.main(x)
