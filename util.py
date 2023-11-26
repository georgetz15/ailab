import random

import numpy as np
import torch


def set_seed(seed=420):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
