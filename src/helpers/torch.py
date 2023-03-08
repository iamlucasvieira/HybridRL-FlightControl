"""Module that contains helper functions for PyTorch."""

import torch as th
from torch import nn


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
