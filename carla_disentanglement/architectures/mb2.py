from numpy.lib.arraysetops import isin
import torch
from torch import nn
from torch import Tensor
from typing import Callable, Optional, List
from .base import Flatten3D, View


class ConvBNActivation(nn.Sequential):
    '''
    https://github.com/pytorch/vision/blob/183a722169421c83638e68ee2d8fc5bd3415c4b4/torchvision/models/mobilenetv2.py
    '''

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
        conv_layer: nn.Module = nn.Conv2d
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU
        super().__init__(
            conv_layer(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                       bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )
        self.out_channels = out_planes


BN_EPS = 1e-5
BN_MOM = 0.05


class SE(nn.Module):
    '''
    https://github.com/NVlabs/NVAE/blob/38eb9977aa6859c6ee037af370071f104c592695/neural_operations.py#L282
    '''

    def __init__(self, Cin, Cout):
        super(SE, self).__init__()
        num_hidden = max(Cout // 16, 4)
        self.se = nn.Sequential(nn.Linear(Cin, num_hidden), nn.ReLU(inplace=True),
                                nn.Linear(num_hidden, Cout), nn.Sigmoid())

    def forward(self, x):
        se = torch.mean(x, dim=[2, 3])
        se = se.view(se.size(0), -1)
        se = self.se(se)
        se = se.view(se.size(0), -1, 1, 1)
        return x * se


class InvertedResidual(nn.Module):
    '''
    Base: https://github.com/pytorch/vision/blob/183a722169421c83638e68ee2d8fc5bd3415c4b4/torchvision/models/mobilenetv2.py
    adapted with ideas from https://github.com/NVlabs/NVAE/blob/38eb9977aa6859c6ee037af370071f104c592695/neural_operations.py#L297
    '''

    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: nn.Module = nn.Conv2d
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = [norm_layer(inp)]
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, conv_layer=conv_layer))
        layers.extend([
            # dw
            ConvBNActivation(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer, conv_layer=conv_layer),
            # pw-linear
            conv_layer(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
            SE(oup, oup)
        ])

        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class BaseEncDec(nn.Module):
    def __init__(self):
        super(BaseEncDec, self).__init__()
        self._image_size = 64

    def init_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


class Encoder(BaseEncDec):
    def __init__(
        self,
        num_latents: int,
        num_channels: int = 3,
        width_mult: int = 1,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        """
        MobileNet V2 based Encoder for 64x64 images
        Args:
            num_latents (int): Number of latents to encode into
            num_channels (int): Number of image input channels
            width_mult (int): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
        """
        super(Encoder, self).__init__()

        self._num_latents = num_latents
        self._num_channels = num_channels

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            def norm_layer(num_f):
                return nn.BatchNorm2d(num_f, eps=BN_EPS, momentum=BN_MOM)

        input_channel = 32

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                # t: expand factor
                # c: number of output channels
                # n: number of Residual blocks in sequence
                # s: stride for *last* block in sequnece (rest stride 1)
                # input image size 32
                [1, 16, 1, 1],  # 32
                [6, 24, 2, 2],  # 16
                [6, 32, 3, 2],  # 8
                [6, 64, 4, 2],  # 4
                # [6, 96, 3, 1],
                # [6, 160, 3, 2],
                # [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = input_channel * width_mult
        # self.last_channel = last_channel * width_mult

        features: List[nn.Module] = [ConvBNActivation(num_channels, input_channel, stride=2, norm_layer=norm_layer)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = c * width_mult
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        # features.append(ConvBNActivation(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        features.append(Flatten3D())
        features.append(nn.Linear(output_channel*4*4, num_latents))

        # make it nn.Sequential
        self.main = nn.Sequential(*features)

        # weight initialization
        self.init_layers()

    def forward(self, x: Tensor) -> Tensor:
        return self.main(x)


class GaussianEncoder(Encoder):
    """
        MobileNet V2 based gaussian Encoder for 64x64 images
        Args:
            num_latents (int): Number of latents to encode into
            num_channels (int): Number of image input channels
            width_mult (int): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
        """

    def __init__(
        self,
        num_latents: int,
        num_channels: int = 3,
        width_mult: int = 1,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ):
        super().__init__(num_latents * 2, num_channels, width_mult, inverted_residual_setting, block, norm_layer)

        self._latent_dim = num_latents

    def forward(self, x):
        mu_logvar = self.main(x)
        mu = mu_logvar[:, :self._latent_dim]
        logvar = mu_logvar[:, self._latent_dim:]
        return mu, logvar


class Decoder(BaseEncDec):
    def __init__(
        self,
        num_latents: int,
        num_channels: int = 3,
        width_mult: int = 1,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        """
        MobileNet V2 based Decoder for 64x64 images
        Args:
            num_latents (int): Number of latents to encode into
            num_channels (int): Number of image input channels
            width_mult (int): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
        """
        super(Decoder, self).__init__()

        self._num_latents = num_latents
        self._num_channels = num_channels

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            def norm_layer(num_f):
                return nn.BatchNorm2d(num_f, eps=BN_EPS, momentum=BN_MOM)

        conv_layer = nn.ConvTranspose2d

        def conv_layer(*args, **kwargs):
            kwargs['output_padding'] = (kwargs.get('stride') or args[3]) - 1
            return nn.ConvTranspose2d(*args, **kwargs)

        input_channel = 32

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                # t: expand factor
                # c: number of output channels
                # n: number of Residual blocks in sequence
                # s: stride for first block in sequnece (rest stride 1)
                # input image size 4
                [6, 64, 4, 2],  # 4
                [6, 32, 3, 2],  # 8
                [6, 24, 2, 2],  # 16
                [1, 16, 1, 1],  # 32
                # [6, 96, 3, 1],
                # [6, 160, 3, 2],
                # [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = inverted_residual_setting[0][1] * width_mult
        # self.last_channel = last_channel * width_mult

        features: List[nn.Module] = [
            nn.Linear(num_latents, input_channel*4*4),
            View((-1, input_channel, 4, 4))
        ]

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = c * width_mult
            for i in range(n):
                stride = s if i == n-1 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer, conv_layer=conv_layer))
                input_channel = output_channel

        # building last several layers
        # features.append(ConvBNActivation(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))

        features.append(ConvBNActivation(input_channel, num_channels, stride=2, norm_layer=norm_layer, conv_layer=conv_layer))

        # make it nn.Sequential
        self.main = nn.Sequential(*features)

        # weight initialization
        self.init_layers()

    def forward(self, x: Tensor) -> Tensor:
        return self.main(x)
