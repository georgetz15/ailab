import random

from torch import nn


class RandomPatch(nn.Module):
    def __init__(self,
                 p=0.5,
                 width=(2, 10),
                 height=(2, 10), ):
        super().__init__()

        self.p = p
        self.width = width
        self.height = height

    def forward(self, x):
        c, h, w = x.shape

        wp, hp = round(random.uniform(*self.width)), round(random.uniform(*self.height))
        xp, yp = round(random.uniform(0, w - wp)), round(random.uniform(0, h - hp))  # patch start
        xf, yf = round(random.uniform(0, w - wp)), round(random.uniform(0, h - hp))  # fill start

        x[:, yp:yp + hp, xp:xp + wp] = x[:, yf:yf + hp, xf:xf + wp]
        return x
