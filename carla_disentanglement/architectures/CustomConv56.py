from .base import BaseImageEncoder, Flatten3D, Unsqueeze3D
from torch import nn


class CustomConv56(BaseImageEncoder):
    def __init__(self, hidden_size, latent_dim, num_channels, image_size):
        super().__init__(latent_dim, num_channels, image_size)
        assert image_size == 56, 'This model only works with image size 56x56.'
        assert (hidden_size % 8) == 0, 'Hidden size needs to be multiple of 8'

        self.main = nn.Sequential(
            nn.Conv2d(num_channels, hidden_size//8, 4, 2, 1),  # 28x28
            nn.ReLU(True),
            nn.Conv2d(hidden_size//8, hidden_size//8, 4, 2, 1),  # 14x14
            nn.ReLU(True),
            nn.Conv2d(hidden_size//8, hidden_size//4, 4, 2, 1),  # 7x7
            nn.ReLU(True),
            nn.Conv2d(hidden_size//4, hidden_size//2, 4, 1, 0),  # 4x4
            nn.ReLU(True),
            nn.Conv2d(hidden_size//2, hidden_size, 4, 2, 1),  # 2x2
            nn.ReLU(True),
            nn.Conv2d(hidden_size, hidden_size, 4, 2, 1),  # 2x2
            nn.ReLU(True),
            Flatten3D(),
            nn.Linear(hidden_size, latent_dim, bias=True)
        )

        self.init_layers()

    def forward(self, x):
        return self.main(x)


class CustomGaussianConv56(CustomConv56):
    def __init__(self, hidden_size, latent_dim, num_channels, image_size):
        super().__init__(hidden_size, latent_dim * 2, num_channels, image_size)

        # override value of _latent_dim
        self._latent_dim = latent_dim

    def forward(self, x):
        mu_logvar = self.main(x)
        mu = mu_logvar[:, :self._latent_dim]
        logvar = mu_logvar[:, self._latent_dim:]
        return mu, logvar


class CustomConv56Decoder(BaseImageEncoder):
    def __init__(self, hidden_size, latent_dim, num_channels, image_size):
        super().__init__(latent_dim, num_channels, image_size)
        assert image_size == 56, 'This model only works with image size 56x56.'
        assert (hidden_size % 8) == 0, 'Hidden size needs to be multiple of 8'

        self.main = nn.Sequential(
            Unsqueeze3D(),
            nn.Conv2d(latent_dim, hidden_size, 1, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size, hidden_size, 4, 2, 1),  # 2x2
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size, hidden_size//2, 4, 2),  # 6x6
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size//2, hidden_size//2, 4, 2, 1),  # 12x12
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size//2, hidden_size//4, 4, 2),  # 26x26
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size//4, hidden_size//4, 4, 2),  # 54x54
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_size//4, num_channels, 3, 1)  # 56x56
        )

        self.init_layers()

    def forward(self, x):
        return self.main(x)
