from collections import defaultdict

import numpy as np
import torch
from torch import nn, optim

from hooks import KeepActivations
from layers import ResNetBlock
import lightning as pl


def get_resnet(act=nn.LeakyReLU, n_features=(8, 16, 32, 64, 128, 256), n_classes=10, norm=nn.BatchNorm2d,
               dropout_rate=0.5):
    layers = [ResNetBlock(1, 8, stride=1, activation=act, norm=norm), ]
    layers += [ResNetBlock(n_features[i], n_features[i + 1], stride=2, norm=norm, activation=act) for i in
               range(len(n_features) - 1)]
    layers += [nn.Flatten(), nn.Dropout(p=dropout_rate, inplace=True), nn.Linear(n_features[-1], n_classes, bias=False),
               nn.BatchNorm1d(n_classes)]
    return nn.Sequential(*layers)


class ResNetClassifier(pl.LightningModule):
    def __init__(self,
                 n_classes,
                 n_features=(8, 16, 32, 64, 128, 256),
                 dropout=0.5,
                 opt="AdamW",
                 lr=1e-2,
                 wd=1e-2, ):
        super().__init__()
        self.classifier = get_resnet(nn.LeakyReLU, n_features, n_classes, nn.BatchNorm2d, dropout, )

        self.save_hyperparameters()

        self._activations_hook = KeepActivations()
        self.__handles = []

    def forward(self, x):
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        self._activations_hook.reset()

        x, y = batch

        preds = self.forward(x)
        loss = nn.functional.cross_entropy(preds, y)

        accuracy = (preds.argmax(-1) == y).float().mean().item()

        for i, act in enumerate(self._activations_hook.activations):
            self.logger.experiment.add_histogram(f'feature_activations/{i}', act, self.global_step)

        self.log_dict({"loss/train": loss}, on_step=False, on_epoch=True)
        self.log_dict({"accuracy/train": accuracy}, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch

        preds = self.forward(x)
        loss = nn.functional.cross_entropy(preds, y)

        accuracy = (preds.argmax(-1) == y).float().mean().item()
        self.log_dict({"loss/val": loss}, on_step=False, on_epoch=True)
        self.log_dict({"accuracy/val": accuracy}, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_start(self):
        pass

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_start(self):
        pass

    def on_validation_epoch_end(self):
        pass

    def configure_optimizers(self):
        if self.hparams.opt == "Adam":
            opt = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        elif self.hparams.opt == "AdamW":
            opt = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        elif self.hparams.opt == "SGD":
            opt = optim.SGD(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd, momentum=0.9)
        elif self.hparams.opt == "RMSprop":
            opt = optim.RMSprop(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        elif self.hparams.opt == "RAdam":
            opt = optim.RAdam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        else:
            raise ValueError

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.hparams.lr, total_steps=self.trainer.estimated_stepping_batches
        )

        return [opt], [scheduler]

    def on_fit_start(self):
        tb = self.logger.experiment  # noqa

        layout_scalar_names_losses = [r"loss/train", "loss/val"]
        layout = {
            self.__class__.__name__: {
                f"loss": ["Multiline", layout_scalar_names_losses],
                f"accuracy": ["Multiline", ["accuracy/train", "accuracy/val"]],
            }
        }

        tb.add_custom_scalars(layout)

        for m in self.classifier.modules():
            if isinstance(m, (ResNetBlock, nn.Linear)):
                handle = m.register_forward_hook(self._activations_hook.forward_hook)
                self.__handles.append(handle)

    def on_fit_end(self) -> None:
        while self.__handles:
            self.__handles.pop().remove()