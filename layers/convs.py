import torch
from torch import nn


def conv3x3(in_channels: int,
            out_channels: int,
            stride: int = 1,
            groups: int = 1,
            dilation: int = 1,
            bias=False,
            **kwargs) -> nn.Conv2d:
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     dilation=dilation,
                     bias=bias,
                     **kwargs)


def sepconv3x3(in_channels: int,
               out_channels: int,
               stride: int = 1,
               groups: int = 1,
               dilation: int = 1,
               bias=False,
               **kwargs) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels,
                  in_channels,
                  kernel_size=(1, 3),
                  stride=stride,
                  padding=(0, dilation),
                  groups=groups,
                  dilation=(1, dilation),
                  bias=bias,
                  **kwargs),
        nn.Conv2d(in_channels,
                  out_channels,
                  kernel_size=(3, 1),
                  stride=stride,
                  padding=(dilation, 0),
                  groups=groups,
                  dilation=(dilation, 1),
                  bias=bias,
                  **kwargs)
    )


def conv1x1(in_channels: int,
            out_channels: int, ) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
