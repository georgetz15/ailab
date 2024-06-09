import torch.nn as nn

from layers.convs import conv3x3, conv1x1


def _conv(
        in_channels,
        out_channels,
        kernel=(3, 3),
        padding=1,
        stride=1,
        norm=nn.BatchNorm2d,
        activation=nn.LeakyReLU, ):
    layers = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel,
            padding=padding,
            stride=stride),
    ]
    if norm: layers.append(norm(out_channels))
    if activation: layers.append(activation())
    return nn.Sequential(*layers)


def _conv_block(
        in_channels,
        out_channels,
        kernel=(3, 3),
        padding=1,
        stride=1,
        norm=nn.BatchNorm2d,
        activation=nn.LeakyReLU, ):
    layers = [_conv(in_channels, out_channels, kernel, padding, stride=1, norm=norm, activation=activation),
              _conv(out_channels, out_channels, kernel, padding, stride=stride, norm=norm, activation=None)]

    if norm:
        nn.init.constant_(layers[-1][1].weight, 0.)

    return nn.Sequential(*layers)


class ResNetBlock_fastai(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel=(3, 3),
                 padding=1,
                 stride=1,
                 norm=None,
                 activation=nn.LeakyReLU, ):
        super().__init__()

        self.convs = _conv_block(in_channels, out_channels, kernel, padding, stride, norm, activation)
        self.idconv = None if in_channels == out_channels else _conv(in_channels, out_channels, 1, 0, 1, norm,
                                                                     activation)
        self.pool = None if stride == 1 else nn.AvgPool2d(2, ceil_mode=True)
        self.activation = activation()

    def forward(self, x):
        pool = x if not self.pool else self.pool(x)
        idx = pool if not self.idconv else self.idconv(pool)
        convx = self.convs(x)
        y = self.activation(convx + idx)
        return y


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 1):
        super().__init__()

        self.conv1 = conv3x3(in_channels, out_channels, groups=groups)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.conv2 = conv3x3(out_channels, out_channels, groups=groups)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.LeakyReLU(inplace=True)

        self.downsample = nn.Identity() if in_channels == out_channels else conv1x1(in_channels, out_channels)

    def forward(self, x):
        # Conv path
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act1(y)
        y = self.conv2(y)

        # Output path
        y += self.downsample(x)
        y = self.bn2(y)
        y = self.act2(y)

        return y
