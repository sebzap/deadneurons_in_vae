from .base import BaseImageEncoder, Flatten3D, Unsqueeze3D, View
from torch import nn
import numpy as np


# https://github.com/JohanYe/Beta-VAE
class ResidualEncoderBlock(nn.Module):
    # Consider addring gated resnet block instead
    # block_type is a string specifying the structure of the block, where:
    #         a = activation
    #         b = batch norm
    #         c = conv layer
    #         d = dropout.
    # For example, bacd (batchnorm, activation, conv, dropout).
    # TODO: ADDTT uses different number of filters in inner, should we consider that? I've only allowed same currently.

    def __init__(self, c_in, c_out, nonlin=nn.ReLU(), kernel_size=3, block_type=None, dropout=None, stride=2):
        super(ResidualEncoderBlock, self).__init__()

        assert all(c in 'abcd' for c in block_type)
        self.c_in, self.c_out = c_in, c_out
        self.nonlin = nonlin
        self.kernel_size = kernel_size
        self.block_type = block_type
        self.dropout = dropout
        self.stride = stride

        self.pre_conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=self.kernel_size // 2, stride=stride)
        res = []  # Am considering throwing these if statements into separate function
        for character in block_type:
            if character == 'a':
                res.append(nonlin)
            elif character == 'b':
                res.append(nn.BatchNorm2d(c_out))
            elif character == 'c':
                res.append(
                    nn.Conv2d(c_out, c_out, kernel_size=kernel_size, padding=self.kernel_size // 2)
                )
            elif character == 'd':
                res.append(nn.Dropout2d(dropout))
        self.res = nn.Sequential(*res)
        self.post_conv = None  # TODO: Ensure this should not be implemented, consult ADDTT

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.res(x) + x
        if self.post_conv is not None:
            x = self.post_conv(x)
        return x.contiguous()

# https://github.com/JohanYe/Beta-VAE


class ResidualDecoderBlock(nn.Module):
    # Consider addring gated resnet block instead
    # block_type is a string specifying the structure of the block, where:
    #         a = activation
    #         b = batch norm
    #         c = conv layer
    #         d = dropout.
    # For example, bacd (batchnorm, activation, conv, dropout).
    # TODO: ADDTT uses different number of filters in inner, should we consider that? I've only allowed same currently.

    def __init__(self, c_in, c_out, nonlin=nn.ReLU(), kernel_size=3, block_type=None, dropout=None, stride=2):
        super(ResidualDecoderBlock, self).__init__()

        assert all(c in 'abcd' for c in block_type)
        self.c_in, self.c_out = c_in, c_out
        self.nonlin = nonlin
        self.kernel_size = kernel_size
        self.block_type = block_type
        self.dropout = dropout
        self.stride = stride

        self.pre_conv = nn.ConvTranspose2d(
            c_in, c_out, kernel_size=kernel_size, padding=self.kernel_size // 2, stride=stride, output_padding=1)
        res = []  # Am considering throwing these if statements into separate function
        for character in block_type:
            if character == 'a':
                res.append(nonlin)
            elif character == 'b':
                res.append(nn.BatchNorm2d(c_out))
            elif character == 'c':
                res.append(
                    nn.ConvTranspose2d(c_out, c_out, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
                )
            elif character == 'd':
                res.append(nn.Dropout2d(dropout))
        self.res = nn.Sequential(*res)
        self.post_conv = None  # TODO: Ensure this should not be implemented, consult ADDTT

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.res(x) + x
        if self.post_conv is not None:
            x = self.post_conv(x)
        return x.contiguous()


class ResConv64(BaseImageEncoder):
    def __init__(self, latent_dim, num_channels, image_size, block_type='cabd', drop_rate=0.1):
        super().__init__(latent_dim, num_channels, image_size)
        assert image_size == 64, 'This model only works with image size 64x64.'

        # self.conv_out_dim = int((self.img_dim / 2 ** (len(filters))) ** 2 * filters[-1])
        self.conv_out_dim = int((image_size / 2 ** 4) ** 2 * 64)

        self.main = nn.Sequential(
            ResidualEncoderBlock(num_channels, 32, kernel_size=3, block_type=block_type, dropout=drop_rate),
            ResidualEncoderBlock(32, 32, kernel_size=3, block_type=block_type, dropout=drop_rate),
            ResidualEncoderBlock(32, 64, kernel_size=3, block_type=block_type, dropout=drop_rate),
            ResidualEncoderBlock(64, 64, kernel_size=3, block_type=block_type, dropout=drop_rate),
            Flatten3D(),
            nn.Linear(self.conv_out_dim, latent_dim),  # B, z_dim
        )

        self.init_layers()

    def forward(self, x):
        return self.main(x)


class GaussianResConv64(ResConv64):
    def __init__(self, latent_dim, num_channels, image_size, block_type='cabd', drop_rate=0.1):
        super().__init__(latent_dim * 2, num_channels, image_size, block_type=block_type, drop_rate=drop_rate)

        # override value of _latent_dim
        self._latent_dim = latent_dim

    def forward(self, x):
        mu_logvar = self.main(x)
        mu = mu_logvar[:, :self._latent_dim]
        logvar = mu_logvar[:, self._latent_dim:]
        return mu, logvar


class ResConv64Decoder(BaseImageEncoder):
    def __init__(self, latent_dim, num_channels, image_size, block_type='cabd', drop_rate=0.1):
        super().__init__(latent_dim, num_channels, image_size)
        assert image_size == 64, 'This model only works with image size 64x64.'

        self.conv_out_dim = int((image_size / 2 ** 4) ** 2 * 64)

        unflatten_dim = int(np.sqrt(self.conv_out_dim / 64))

        self.main = nn.Sequential(
            nn.Linear(latent_dim, self.conv_out_dim),
            nn.ReLU(),
            View((-1, 64, unflatten_dim, unflatten_dim)),
            ResidualDecoderBlock(64, 64, kernel_size=3, block_type=block_type, dropout=drop_rate),
            ResidualDecoderBlock(64, 32, kernel_size=3, block_type=block_type, dropout=drop_rate),
            ResidualDecoderBlock(32, 32, kernel_size=3, block_type=block_type, dropout=drop_rate),
            ResidualDecoderBlock(32, num_channels, kernel_size=3, block_type=block_type, dropout=drop_rate),
        )

        self.init_layers()

    def forward(self, x):
        return self.main(x)
