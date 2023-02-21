"""Module that defines the IDHP networks."""

import torch as T
import torch.nn as nn
from abc import ABC, abstractmethod

from helpers.misc import get_device


class BaseNetwork(nn.Module, ABC):
    """Base network class."""

    def __init__(self,
                 input_dims=(2,),
                 hidden_dims=256, output_dims=(1,),
                 name='base',
                 chkpt_dir='envs',
                 device=None):
        """Initialize network.

        args:
            beta: Learning rate.
            input_dims: Input dimensions.
            hidden_dims: Number of neurons in the hidden layer.
            name: Name of the network.
            chkpt_dir: Checkpoint directory.
        """
        super(BaseNetwork, self).__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims

        self.model = None

        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = chkpt_dir + f'/{name}_sac'

        self._setup_model()

        self.device = T.device(get_device() if device is None else device)
        self.to(self.device)

    def save_checkpoint(self):
        """Save checkpoint."""
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """Load checkpoint."""
        self.load_state_dict(T.load(self.checkpoint_file))

    @abstractmethod
    def _setup_model(self) -> None:
        """Set up the model."""

    def forward(self, x):
        x = self.flatten(x)
        logits = self.model(x)
        return logits


class CriticNetwork(BaseNetwork):
    """Creates the critic neural network."""

    def __init__(self, input_dims=(2,), hidden_dims=256, output_dims=(1,), name='critic', chkpt_dir='envs'):
        """Initialize critic network.

        args:
            beta: Learning rate.
            input_dims: Input dimensions.
            n_actions: Number of actions.
            fc1_dims: Number of neurons in the first layer.
            fc2_dims_: Number of neurons in the second layer.
        """
        # Input layer of critic's neural network in SAC uses state-action pairs
        super(CriticNetwork, self).__init__(input_dims=input_dims,
                                            hidden_dims=hidden_dims,
                                            output_dims=output_dims,
                                            name=name,
                                            chkpt_dir=chkpt_dir)

    def _setup_model(self) -> None:
        """Set up the model."""

        self.model = nn.Sequential(
            nn.Linear(*self.input_dims, self.hidden_dims),
            nn.Tanh(),
            nn.Linear(self.hidden_dims, *self.output_dims),
        )

        self.optimizer = T.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)


class ActorNetwork(BaseNetwork):
    """Creates the actor network."""

    def __init__(self, input_dims=(2,), hidden_dims=256, output_dims=(1,), name='actor', chkpt_dir='envs'):
        """Initialize actor network.

        args:
            beta: Learning rate.
            input_dims: Input dimensions.
            n_actions: Number of actions.
            fc1_dims: Number of neurons in the first layer.
            fc2_dims_: Number of neurons in the second layer.
        """
        # Input layer of critic's neural network in SAC uses state-action pairs
        super(CriticNetwork, self).__init__(input_dims=input_dims,
                                            hidden_dims=hidden_dims,
                                            output_dims=output_dims,
                                            name=name,
                                            chkpt_dir=chkpt_dir)

    def _setup_model(self) -> None:
        """Set up the model."""

        self.model = nn.Sequential(
            nn.Linear(*self.input_dims, self.hidden_dims),
            nn.Tanh(),
            nn.Linear(self.hidden_dims, *self.output_dims),
            nn.Tanh(),
        )

        self.optimizer = T.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)


bn = CriticNetwork()
