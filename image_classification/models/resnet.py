from collections import defaultdict

import numpy as np
import torch
from torch import nn, optim
from typing import Tuple

from hooks import KeepActivations
from image_classification.models.classification_model import ClassificationModel
from layers import ResNetBlock_fastai
import lightning as pl

from layers.resnet_block import ResnetBlock
from weight_init import generic_init


def get_resnet(act=nn.LeakyReLU,
               n_features=(8, 16, 32, 64, 128, 256),
               n_classes=10,
               norm=nn.BatchNorm2d,
               dropout_rate=0.5):
    layers = [ResNetBlock_fastai(1, n_features[0], stride=1, activation=act, norm=norm), ]
    layers += [ResNetBlock_fastai(n_features[i], n_features[i + 1], stride=2, norm=norm, activation=act)
               for i in range(len(n_features) - 1)]
    layers += [nn.Flatten(), nn.Dropout(p=dropout_rate, inplace=True), nn.Linear(n_features[-1], n_classes, bias=False),
               nn.BatchNorm1d(n_classes)]
    return nn.Sequential(*layers)


class ResNetClassifier_fastai(nn.Module):
    def __init__(self,
                 n_classes,
                 n_features=(8, 16, 32, 64, 128, 256),
                 dropout=0.5, ):
        super().__init__()
        self.classifier = get_resnet(nn.LeakyReLU, n_features, n_classes, nn.BatchNorm2d, dropout, )

    def forward(self, x):
        return self.classifier(x)


class Resnet(nn.Module):
    def __init__(self,
                 n_input_channels: int,
                 n_classes: int,
                 n_features: Tuple[int] = (16, 32, 64, 128, 256),
                 n_hidden_layers: int = 128,
                 dropout: float = 0.5,
                 avg_pool_sz: Tuple[int] = (1, 1),
                 conv_groups: int = 1,
                 init_weights: bool = True, ):
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
            groups = conv_groups if i > 0 else 1

            # Conv block
            features.extend([
                ResnetBlock(n_in, n_out, groups=groups),
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
