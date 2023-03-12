"""Create policy for SAC algorithm."""
import os
import torch as th
from torch import nn
import torch.optim as optim
from torch.distributions import Normal
from torch.nn.functional import softplus
import numpy as np

from typing import Union, List
from gym import spaces
import pathlib as pl
from abc import ABC
from helpers.torch import mlp


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
        self.learning_rate = learning_rate
        self.observation_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]

        self.ff = self._build_network()
        self.name = name

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = device
        self.to(self.device)

    def _build_network(self) -> nn.Sequential:
        """Build network."""
        raise NotImplementedError

    def save_checkpoint(self):
        """Save checkpoint."""
        th.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """Load checkpoint."""
        self.load_state_dict(th.load(self.checkpoint_file))


class CriticNetwork(BaseNetwork):
    """Creates the critic neural network."""

    def __init__(self,
                 observation_space: spaces.Box,
                 action_space: spaces.Box,
                 **kwargs):
        """Initialize critic network.

        args:
            beta: Learning rate.
            input_dims: Input dimensions.
            n_actions: Number of actions.
            fc1_dims: Number of neurons in the first layer.
            fc2_dims_: Number of neurons in the second layer.
        """
        # Input layer of critic's neural network in SAC uses state-action pairs
        super(CriticNetwork, self).__init__(observation_space,
                                            action_space,
                                            **kwargs)

    def _build_network(self) -> nn.Sequential:
        """Build network."""
        ff = mlp([self.observation_dim + self.action_dim] + self.hidden_layers + [1],
                 activation=nn.ReLU)
        return ff

    def forward(self, state, action):
        """Forward pass of the critic's neural network."""
        q = self.ff(th.cat([state, action], dim=-1))
        return th.squeeze(q, -1)


class ActorNetwork(BaseNetwork):
    """Actor network in SAC."""

    def __init__(self,
                 observation_space: spaces.Box,
                 action_space: spaces.Box,
                 sigma_min: float = -30,
                 sigma_max: float = 2,
                 **kwargs):
        """Initialize actor network.

        args:
            observation_space: Observation space.
            action_space: Action space.
            sigma_min: Minimum value of the standard deviation.
            sigma_max: Maximum value of the standard deviation.
        """
        super(ActorNetwork, self).__init__(observation_space,
                                           action_space,
                                           **kwargs)
        self.mu = nn.Linear(self.hidden_layers[-1], self.action_dim)
        self.log_sigma = nn.Linear(self.hidden_layers[-1], self.action_dim)
        self.action_max = float(action_space.high[0])

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def _build_network(self) -> nn.Sequential:
        """Build network."""
        ff = mlp([self.observation_dim] + self.hidden_layers,
                 activation=nn.ReLU,
                 output_activation=nn.ReLU)
        return ff

    def forward(self, state, with_log_prob=True, deterministic=False):
        """Forward pass in the actor network.

        args:
            state: State.
            with_log_prob: Whether to return the log probability.
        """
        net_output = self.ff(state)
        mu = self.mu(net_output)
        log_sigma = th.clamp(self.log_sigma(net_output), min=self.sigma_min, max=self.sigma_max)
        sigma = th.exp(log_sigma)

        action_distribution = Normal(mu, sigma)

        action = action_distribution.rsample() if not deterministic else mu

        if with_log_prob:  # From OpenAi Spinning Up
            log_prob = action_distribution.log_prob(action) - \
                      2 * (np.log(2) - action - softplus(-2 * action))
            log_prob = log_prob.sum(axis=-1)
        else:
            log_prob = None

        action = th.tanh(action) * self.action_max

        return action, log_prob


class SACPolicy(nn.Module):
    """Policy for SAC algorithm."""

    def __init__(self,
                 observation_space: spaces.Box,
                 action_space: spaces.Box,
                 learning_rate: float = 3e-4,
                 hidden_layers: List[int] = None,
                 save_path: Union[str, pl.Path] = ""):
        """Initialize policy.

        args:
            observation_space: Observation space.
            action_space: Action space.
            learning_rate: Learning rate.
            hidden_layers: Number of hidden layers.
            save_path: Path to save the policy.

        """
        super(SACPolicy, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers
        self.save_path = save_path

        self.actor = ActorNetwork(observation_space,
                                  action_space,
                                  learning_rate=learning_rate,
                                  hidden_layers=hidden_layers,
                                  save_path=save_path)
        self.critic_1 = CriticNetwork(observation_space,
                                      action_space,
                                      learning_rate=learning_rate,
                                      hidden_layers=hidden_layers,
                                      save_path=save_path)
        self.critic_2 = CriticNetwork(observation_space,
                                      action_space,
                                      learning_rate=learning_rate,
                                      hidden_layers=hidden_layers,
                                      save_path=save_path)

    def get_action(self, state: np.ndarray, **kwargs) -> th.Tensor:
        """Get action from the policy.

        args:
            state: Current state.

        returns:
            action: Action to take.
        """
        with th.no_grad():
            state = th.tensor(state, dtype=th.float32, device=self.actor.device)
            action, _ = self.actor(state, **kwargs)
        return action.numpy()

    def save(self):
        """Save policy."""
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load(self):
        """Load policy."""
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def predict(self, observation: np.ndarray,
                state: np.ndarray,
                episode_start: np.ndarray,
                deterministic: bool = False) -> np.ndarray:
        """Predict action."""
        # Unused arguments
        del episode_start
        return self.get_action(observation, deterministic=deterministic), state
