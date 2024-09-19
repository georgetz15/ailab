from torch import nn

from layers.convs import conv1x1, depthwise_conv3x3


class MobilenetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int):
        super().__init__()

        self.layer = nn.Sequential(
            conv1x1(in_channels, hidden_channels),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(inplace=True),
            depthwise_conv3x3(hidden_channels),
            nn.BatchNorm2d(hidden_channels),
            conv1x1(hidden_channels, out_channels),
            nn.BatchNorm2d(out_channels),
        )

        self.skip_connection = nn.Sequential(
            conv1x1(in_channels, out_channels)
        )

    def forward(self, x):
        y = self.layer(x)
        y += self.skip_connection(x)

        return y
