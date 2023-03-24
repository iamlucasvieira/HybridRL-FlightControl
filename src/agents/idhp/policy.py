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
        learning_rate: float = 0.08,
        **kwargs
    ):
        """Initialize the base network.

        Args:
            hidden_layers: List of hidden layers.
            learning_rate: Learning rate.
        """
        if hidden_layers is None:
            hidden_layers = [10, 10]

        super().__init__(
            *args, **kwargs, hidden_layers=hidden_layers, learning_rate=learning_rate
        )
        self.flatten = nn.Flatten()
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
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
        )

        return ff

    def get_loss(self, dr1_ds1, gamma, critic_t1, G_t_1):
        """Gets the network loss."""
        return -(dr1_ds1 + gamma * critic_t1) @ G_t_1

    def forward(self, x: Union[np.ndarray, th.Tensor], to_scale: bool = True):
        """Forward pass."""
        action = super().forward(x)
        if to_scale:
            action = scale_action(action, self.action_space) * 10
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

    def predict(self, observation: th.Tensor, deterministic: bool = True):
        """Predict the action."""
        return self.actor(observation)


def scale_action(action: th.tensor, action_space) -> np.ndarray:
    """Scale the action to the correct range."""
    low, high = action_space.low[0], action_space.high[0]
    return action * (high - low) / 2 + (high + low) / 2
