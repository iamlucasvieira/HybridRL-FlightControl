"""Module that contains helper functions for PyTorch."""

import torch as th
from torch import nn
from typing import Union, Tuple


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    """Build a multi-layer perceptron (MLP) with the given sizes and activation functions."""
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [th.nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def get_device():
    """Get device to use for PyTorch."""

    # Check if GPU is available
    has_gpu = th.cuda.is_available()

    # Check if MPS is available
    has_mps = th.backends.mps.is_available()

    # Use MPS if available, otherwise GPU if available, otherwise CPU
    device = "mps" if has_mps else "gpu" if has_gpu else "cpu"

    return device


def to_tensor(*arrays, data_type=th.float32) -> Union[th.Tensor, Tuple[th.Tensor]]:
    """Convert numpy arrays to PyTorch tensors.

    args:
        arrays: Numpy arrays to convert.
        data_type: Data type to use for the tensors.
    """
    tensors = tuple(th.as_tensor(a, dtype=data_type) for a in arrays)
    if len(tensors) == 1:
        return tensors[0]
    return tensors


def update_requires_grad(module: nn.Module, requires_grad: bool):
    """Update the requires_grad attribute of the parameters of the given module."""
    for param in module.parameters():
        param.requires_grad = requires_grad


def freeze(module: nn.Module):
    """Freeze the parameters of the given module."""
    update_requires_grad(module, False)


def unfreeze(module: nn.Module):
    """Unfreeze the parameters of the given module."""
    update_requires_grad(module, True)
