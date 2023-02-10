from torch import nn
import torch.nn.init as init


def _init_layer(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


class BaseImageEncoder(nn.Module):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__()

        self._latent_dim = latent_dim
        self._num_channels = num_channels
        self._image_size = image_size

    def forward(self, *input):
        raise NotImplementedError

    def latent_dim(self):
        return self._latent_dim

    def num_channels(self):
        return self._num_channels

    def image_size(self):
        return self._image_size

    def init_layers(self):
        for block in self._modules:
            from collections.abc import Iterable
            if isinstance(self._modules[block], Iterable):
                for m in self._modules[block]:
                    _init_layer(m)
            else:
                _init_layer(self._modules[block])


class Flatten3D(nn.Module):
    def forward(self, x):
        x = x.reshape(x.size()[0], -1)
        return x


class Unsqueeze3D(nn.Module):
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = x.unsqueeze(-1)
        return x


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
