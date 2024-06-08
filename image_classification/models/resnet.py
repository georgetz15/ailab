from collections import defaultdict

import numpy as np
import torch
from torch import nn, optim

from hooks import KeepActivations
from image_classification.models.classification_model import ClassificationModel
from layers import ResNetBlock
import lightning as pl


def get_resnet(act=nn.LeakyReLU,
               n_features=(8, 16, 32, 64, 128, 256),
               n_classes=10,
               norm=nn.BatchNorm2d,
               dropout_rate=0.5):
    layers = [ResNetBlock(1, n_features[0], stride=1, activation=act, norm=norm), ]
    layers += [ResNetBlock(n_features[i], n_features[i + 1], stride=2, norm=norm, activation=act)
               for i in range(len(n_features) - 1)]
    layers += [nn.Flatten(), nn.Dropout(p=dropout_rate, inplace=True), nn.Linear(n_features[-1], n_classes, bias=False),
               nn.BatchNorm1d(n_classes)]
    return nn.Sequential(*layers)


class ResNetClassifier(nn.Module):
    def __init__(self,
                 n_classes,
                 n_features=(8, 16, 32, 64, 128, 256),
                 dropout=0.5, ):
        super().__init__()
        self.classifier = get_resnet(nn.LeakyReLU, n_features, n_classes, nn.BatchNorm2d, dropout, )

    def forward(self, x):
        return self.classifier(x)
