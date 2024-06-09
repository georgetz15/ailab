from enum import Enum

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


class NonLinearityType(Enum):
    LEAKY_RELU = "leaky_relu"
    RELU = "relu"


def generic_init(model: nn.Module,
                 nonlinearity: NonLinearityType = NonLinearityType.LEAKY_RELU,
                 init_args=None):
    if model is None:
        return

    # Validations
    if nonlinearity not in {NonLinearityType.LEAKY_RELU, NonLinearityType.RELU}:
        raise NotImplemented(f"Initialisation not implemented for {nonlinearity} activations")
    if init_args is None:
        init_args = dict()
        init_args["a"] = 1e-2 if nonlinearity == NonLinearityType.LEAKY_RELU else 0

    # Perform initialization
    if isinstance(model, nn.Conv2d):
        nn.init.kaiming_normal_(model.weight, mode="fan_out", nonlinearity=nonlinearity.value, **init_args)
        if model.bias is not None:
            nn.init.constant_(model.bias, 0)
    elif isinstance(model, nn.BatchNorm2d):
        nn.init.constant_(model.weight, 1)
        nn.init.constant_(model.bias, 0)
    elif isinstance(model, nn.Linear):
        nn.init.normal_(model.weight, 0, 0.01)
        nn.init.constant_(model.bias, 0)
