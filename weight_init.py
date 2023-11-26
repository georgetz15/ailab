from torch import nn


def init_for_relu(model, a=0.01, apply_to=(nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
    """
    Apply initialization for ReLU or LeakyReLU layers. Use with `model.apply(init_for_relu)`
    :param model: torch.nn.Module
    :param a: the leaky relu weight
    :param apply_to: a tuple of layers to apply initialization to
    """
    if isinstance(model, apply_to):
        nn.init.kaiming_normal_(model.weight, a=a)


def init_for_sigmoid(model, apply_to=(nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
    """
    Apply initialization for Sigmoid layers. Use with `model.apply(init_for_sigmoid)`
    :param model: torch.nn.Module
    :param a: the leaky relu weight
    :param apply_to: a tuple of layers to apply initialization to
    """
    if isinstance(model, apply_to):
        nn.init.xavier_normal_(model.weight)
