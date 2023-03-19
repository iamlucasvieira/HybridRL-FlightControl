"""Module that contains helper functions for PyTorch."""

import pathlib as pl
from abc import ABC, abstractmethod
from typing import Union, Tuple

import numpy as np
import torch as th
import torch.optim as optim
from gymnasium import spaces
from torch import nn


class BaseNetwork(nn.Module, ABC):
    """Base network for Critic and Value networks."""

    def __init__(self,
                 observation_space: spaces.Box,
                 action_space: spaces.Box,
                 learning_rate: float = 3e-4,
                 hidden_layers=None,
                 name='base',
                 save_path: Union[str, pl.Path] = "",
                 device: Union[str, th.device] = "cpu"):
        """Initialize critic network.

        args:
            beta: Learning rate.
            input_dims: Input dimensions.
            fc1_dims: Number of neurons in the first layer.
            fc2_dims_: Number of neurons in the second layer.
            name: Name of the network.
            chkpt_dir: Checkpoint directory.

        """
        super(BaseNetwork, self).__init__()
        if hidden_layers is None:
            hidden_layers = [256, 256]

        self.hidden_layers = hidden_layers
        self.num_hidden_layers = len(hidden_layers)
        self.learning_rate = learning_rate
        self.observation_dim = observation_space.shape[0]
        self.observation_space = observation_space
        self.action_dim = action_space.shape[0]
        self.action_space = action_space

        self.ff = self._build_network()
        self.name = name

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = device
        self.to(self.device)

    @abstractmethod
    def _build_network(self) -> nn.Sequential:
        """Build network."""
        raise NotImplementedError

    def save_checkpoint(self):
        """Save checkpoint."""
        th.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """Load checkpoint."""
        self.load_state_dict(th.load(self.checkpoint_file))


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity, bias=True):
    """Build a multi-layer perceptron (MLP) with the given sizes and activation functions."""
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [th.nn.Linear(sizes[j], sizes[j + 1], bias=bias), act()]
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

    def _to_array(a):
        if isinstance(a, np.ndarray or tuple):
            return a
        return np.array(a)

    tensors = tuple(th.as_tensor(_to_array(a), dtype=data_type) for a in arrays)
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
