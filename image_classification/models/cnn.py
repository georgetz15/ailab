import torch
import torch.nn as nn
from typing import Tuple, Optional

from weight_init import generic_init


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


def conv_block(in_channels: int,
               out_channels: int,
               norm: Optional[nn.Module] = None,
               activation: nn.Module = nn.LeakyReLU()):
    layers = [
        conv3x3(in_channels, out_channels),
        norm if norm else nn.Identity(),
        activation,
    ]
    return nn.Sequential(*layers)


class CNN(nn.Module):
    def __init__(self,
                 n_classes: int,
                 n_features: Tuple[int],
                 n_hidden_layers: int,
                 n_input_channels: int = 1,
                 dropout: float = 0.5,
                 avg_pool_sz: Tuple[int] = (1, 1),
                 init_weights: bool = True,
                 use_sepconv: bool = False, ):
        # Initialization and validation
        super().__init__()
        if len(n_features) < 1:
            raise ValueError(f"n_features should contain at least 1 element, but has len(n_features)={len(n_features)}")
        if any(n for n in n_features if n < 1):
            raise ValueError(f"n_features should contain only positive number of features.")

        # Create the feature extractor
        n_layers = len(n_features)
        features = []
        for i in range(n_layers - 1):
            n_in = n_features[i] if i > 0 else n_input_channels
            n_out = n_features[i + 1]

            # Conv block
            features.extend([
                conv3x3(n_in, n_out) if not use_sepconv else sepconv3x3(n_in, n_out),
                nn.BatchNorm2d(n_out),
                nn.LeakyReLU(inplace=True),
            ])
            # Max pool
            if i < n_layers - 1:
                features.append(nn.MaxPool2d(2, 2))
        self.features = nn.Sequential(*features)

        # Average pooling to make the extracted features compatible with classifier regardless of image size
        self.avgpool = nn.AdaptiveAvgPool2d(avg_pool_sz)

        # Classifier setup
        self.classifier = nn.Sequential(
            nn.Linear(n_features[-1] * avg_pool_sz[0] * avg_pool_sz[1], n_hidden_layers),
            nn.LeakyReLU(True),
            nn.Dropout(dropout),
            nn.Linear(n_hidden_layers, n_classes),
        )

        # Init weights
        if init_weights:
            self.apply(generic_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.features(x)
        y = self.avgpool(y)
        y = torch.flatten(y, 1)
        y = self.classifier(y)
        return y
