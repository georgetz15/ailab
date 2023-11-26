import torch.nn as nn


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


class ResNetBlock(nn.Module):
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
