"""Module that defines the IDHP agent, including the actor and the critic."""
from typing import Type, List

import gym
import torch
import torch.nn as nn
import torch.optim as optim

from helpers.torch import mlp, BaseNetwork


class BaseNetworkIDHP(BaseNetwork):
    """Base network class for the IDHP agent."""

    def __init__(self,
                 *args,
                 hidden_layers: List[int] = None,
                 learning_rate: float = 0.08,
                 **kwargs):
        """Initialize the base network.

        Args:
            hidden_layers: List of hidden layers.
            learning_rate: Learning rate.
        """
        if hidden_layers is None:
            hidden_layers = [10, 10]

        super(BaseNetworkIDHP, self).__init__(*args, **kwargs,
                                              hidden_layers=hidden_layers,
                                              learning_rate=learning_rate
                                              )
        self.flatten = nn.Flatten()
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        self.device = 'cpu'  # Having issues not using cpu with get_device()
        self.to(self.device)

    def forward(self, x):
        """Forward pass."""
        # flatten if needed
        if len(x.shape) > 1:
            x = self.flatten(x)

        # Transform into tensor if needed
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        logits = self.ff(x)
        return logits


class Actor(BaseNetworkIDHP):
    """Class that implements the actor network for the IDHP agent."""

    def __init__(self,
                 *args,
                 **kwargs):
        """Initialize the actor network."""
        super(Actor, self).__init__(*args, **kwargs)

    def _build_network(self) -> Type[nn.Sequential]:
        """Build the network."""
        ff = mlp([self.observation_space.shape[0]] + self.hidden_layers + [self.action_space.shape[0]],
                 activation=nn.Tanh,
                 output_activation=nn.Tanh,
                 bias=False)
        return ff

    def get_loss(self, dr1_ds1, gamma, critic_t1, G_t_1):
        """Gets the network loss."""
        return -(dr1_ds1 + gamma * critic_t1) @ G_t_1


class Critic(BaseNetworkIDHP):
    """Class that implements the critic network for the IDHP agent."""

    def __init__(self,
                 *args,
                 **kwargs):
        """Initialize the critic network."""
        super(Critic, self).__init__(*args, **kwargs)

    def _build_network(self) -> Type[nn.Sequential]:
        """Build the network."""
        # Minus 1 in the observation states to include only the states and not the reference signal
        ff = mlp([self.observation_space.shape[0]] + self.hidden_layers + [self.observation_space.shape[0] - 1],
                 activation=nn.Tanh,
                 bias=False)
        return ff

    def get_loss(self, dr1_ds1, gamma, critic_t, critic_t1, F_t_1, G_t_1, obs_grad):
        """Gets the network loss."""
        return critic_t - (dr1_ds1 + gamma * critic_t1) @ (F_t_1 + G_t_1 @ obs_grad[:, :-1])


class IDHPPolicy:
    """Class that implements the IDHP policy."""

    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 actor_kwargs={},
                 critic_kwargs={}, ):
        """Initialize the IDHP policy."""

        self.actor = Actor(observation_space, action_space, **actor_kwargs)
        self.critic = Critic(observation_space, action_space, **critic_kwargs)

    def predict(self, observation, state=None, episode_start=None, deterministic=None):
        """Predict the action."""
        return self.actor(observation), observation
