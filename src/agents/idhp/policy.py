"""Module that defines the IDHP agent, including the actor and the critic."""
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from typing import Type
from abc import ABC, abstractmethod
from helpers.misc import get_device


class BaseNetwork(nn.Module, ABC):
    """Base network class for the IDHP agent.

    Attributes:
        observation_space: The observation space of the environment.
        action_space: The action space of the environment.
        hidden_size: The size of the hidden layers.
        num_layers: The number of hidden layers.
    """

    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 hidden_size: int = 10,
                 num_layers: int = 2,
                 learning_rate: float = 0.08):
        """Initialize the base network.

        Args:
            observation_space: The observation space of the environment.
            action_space: The action space of the environment.
            hidden_size: The size of the hidden layers.
            num_layers: The number of hidden layers.
            learning_rate: The learning rate.
        """
        super(BaseNetwork, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.flatten = nn.Flatten()
        self.ff = self._build_network()

        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        self.device = 'cpu'  # Having issues not using cpu with get_device()
        self.to(self.device)

    @abstractmethod
    def _build_network(self) -> Type[nn.Sequential]:
        """Build the network."""
        pass

    @abstractmethod
    def get_loss(self):
        """Gets the network loss."""
        pass

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


class Actor(BaseNetwork):
    """Class that implements the actor network for the IDHP agent."""

    def __init__(self,
                 *args,
                 **kwargs):
        """Initialize the actor network."""
        super(Actor, self).__init__(*args, **kwargs)

    def _build_network(self) -> Type[nn.Sequential]:
        """Build the network."""
        ff = nn.Sequential(
            nn.Linear(self.observation_space.shape[0], self.hidden_size, bias=False),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size, bias=False),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.action_space.shape[0], bias=False),
            nn.Tanh(),
        )
        return ff

    def get_loss(self, dr1_ds1, gamma, critic_t1, G_t_1):
        """Gets the network loss."""
        return -(dr1_ds1 + gamma * critic_t1) @ G_t_1


class Critic(BaseNetwork):
    """Class that implements the critic network for the IDHP agent."""

    def __init__(self,
                 *args,
                 **kwargs):
        """Initialize the critic network."""
        super(Critic, self).__init__(*args, **kwargs)

    def _build_network(self) -> Type[nn.Sequential]:
        """Build the network."""
        ff = nn.Sequential(
            nn.Linear(self.observation_space.shape[0], self.hidden_size, bias=False),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size, bias=False),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.observation_space.shape[0] - 1, bias=False),
            # Minus 1 to include only the states
        )
        return ff

    def get_loss(self, dr1_ds1, gamma, critic_t, critic_t1, F_t_1, G_t_1, obs_grad):
        """Gets the network loss."""
        return critic_t - (dr1_ds1 + gamma * critic_t1) @ (F_t_1 + G_t_1 @ obs_grad[:, :-1])


class IDHPPolicy:
    """Class that implements the IDHP policy."""

    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 **kwargs):
        """Initialize the IDHP policy."""
        self.actor = Actor(observation_space, action_space, **kwargs)
        self.critic = Critic(observation_space, action_space, **kwargs)

    def predict(self, observation, state=None, episode_start=None, deterministic=None):
        """Predict the action."""
        return self.actor(observation), observation
