import numpy as np
import torch
from torch import nn, optim

from collections import defaultdict

from hooks import KeepActivations
import lightning as pl


class MLPClassifier(nn.Module):
    def __init__(self,
                 input_sz,
                 n_classes,
                 n_features=(16, 32,),
                 dropout=0.5, ):
        assert len(n_features) >= 1
        assert n_classes > 0
        assert 0 <= dropout <= 1

        super().__init__()

        layers = [
            nn.Flatten(),
            nn.Linear(input_sz, n_features[0]),
            nn.BatchNorm1d(n_features[0]),
            nn.LeakyReLU(),
        ]
        for i in range(1, len(n_features)):
            layers.extend([
                nn.Linear(n_features[i - 1], n_features[i]),
                nn.BatchNorm1d(n_features[i]),
                nn.LeakyReLU()
            ])
        layers.extend([
            nn.Dropout(dropout),
            nn.Linear(n_features[-1], n_classes),
        ])

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)
