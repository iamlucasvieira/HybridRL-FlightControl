"""Module that defines the IDHP agent, including the actor and the critic."""
from typing import List, Optional, Type, Union

import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces

from agents import BasePolicy
from helpers.torch_helpers import BaseNetwork, mlp


class BaseNetworkIDHP(BaseNetwork):
    """Base network class for the IDHP agent."""

    def __init__(
        self,
        *args,
        hidden_layers: List[int] = None,
        lr_low: float = 0.005,
        lr_high: float = 0.08,
        lr_threshold: float = 1,
        **kwargs
    ):
        """Initialize the base network.

        Args:
            hidden_layers: List of hidden layers.
            learning_rate: Learning rate.
        """
        if hidden_layers is None:
            hidden_layers = [10, 10]

        self.lr_low = lr_low
        self.lr_high = lr_high
        self.lr_threshold = lr_threshold

        super().__init__(
            *args, **kwargs, hidden_layers=hidden_layers, learning_rate=lr_high
        )
        self.flatten = nn.Flatten()
        self.optimizer = optim.SGD(self.parameters(), lr=lr_high)
        # self.device = "cpu"  # Having issues not using cpu with get_device()
        self.to(self.device)

    def forward(self, x):
        """Forward pass."""
        # flatten if needed
        if len(x.shape) > 1:
            x = self.flatten(x)

        # Transform into tensor if needed
        if not isinstance(x, th.Tensor):
            x = th.tensor(x, dtype=th.float32, device=self.device)
        return self.ff(x)

    def update_learning_rate(self, loss):
        """Update the learning rate."""
        if loss < self.lr_threshold:
            self.optimizer.param_groups[0]["lr"] = self.lr_low
        else:
            self.optimizer.param_groups[0]["lr"] = self.lr_high


class Actor(BaseNetworkIDHP):
    """Class that implements the actor network for the IDHP agent."""

    def __init__(self, *args, **kwargs):
        """Initialize the actor network."""
        super().__init__(*args, **kwargs)

    def _build_network(self) -> Type[nn.Sequential]:
        """Build the network."""
        ff = mlp(
            [self.observation_space.shape[0]]
            + self.hidden_layers
            + [self.action_space.shape[0]],
            activation=nn.Tanh,
            output_activation=nn.Tanh,
            bias=False,
            layer_norm=False,
        )

        return ff

    def get_loss(self, dr1_ds1, gamma, critic_t1, G_t_1):
        """Gets the network loss."""
        return -(dr1_ds1 + gamma * critic_t1) @ G_t_1

    def forward(self, x: Union[np.ndarray, th.Tensor]):
        """Forward pass."""
        action = super().forward(x)
        return action


class Critic(BaseNetworkIDHP):
    """Class that implements the critic network for the IDHP agent."""

    def __init__(self, *args, **kwargs):
        """Initialize the critic network."""
        super().__init__(*args, **kwargs)

    def _build_network(self) -> Type[nn.Sequential]:
        """Build the network."""
        # Minus 1 in the observation states to include only the states and not the reference signal
        ff = mlp(
            [self.observation_space.shape[0]]
            + self.hidden_layers
            + [self.observation_space.shape[0] - 1],
            activation=nn.Tanh,
            bias=False,
        )
        return ff

    def get_loss(self, dr1_ds1, gamma, critic_t, critic_t1, F_t_1, G_t_1, obs_grad):
        """Gets the network loss."""
        return critic_t - (dr1_ds1 + gamma * critic_t1) @ (
            F_t_1 + G_t_1 @ obs_grad[:, :-1]
        )


class IDHPPolicy(BasePolicy):
    """Class that implements the IDHP policy."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        actor_kwargs: Optional[dict] = None,
        critic_kwargs: Optional[dict] = None,
        device: Optional[str] = None,
    ):
        """Initialize the IDHP policy."""
        self.actor_kwargs = {} if actor_kwargs is None else actor_kwargs
        self.critic_kwargs = {} if critic_kwargs is None else critic_kwargs
        super().__init__(observation_space, action_space, device=device)

    def _setup_policy(self):
        """Set up the policy."""
        self.actor = Actor(
            self.observation_space,
            self.action_space,
            device=self.device,
            **self.actor_kwargs
        )
        self.critic = Critic(
            self.observation_space,
            self.action_space,
            device=self.device,
            **self.critic_kwargs
        )

    def _predict(self, observation: th.Tensor, deterministic: bool = True):
        """Predict the action."""
        return self.actor(observation)
