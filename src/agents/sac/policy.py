"""Create policy for SAC algorithm."""
import os
import torch as th
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

from typing import Union, List
from gym import spaces
import pathlib as pl
from abc import ABC
from helpers.torch import mlp


class BaseNetwork(nn.Module, ABC):
    """Base network for Critic and Value networks."""

    def __init__(self,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
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
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
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
        return self.ff(th.cat([state, action], dim=1))


class ActorNetwork(BaseNetwork):
    """Actor network in SAC."""

    def __init__(self,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 **kwargs):
        """Initialize actor network.

        args:
            alpha: Learning rate.
            input_dims: Input dimensions.
            fc1_dims: Number of neurons in the first layer.
            fc2_dims_: Number of neurons in the second layer.
            name: Name of the network.
            chkpt_dir: Checkpoint directory.
        """
        super(ActorNetwork, self).__init__(observation_space,
                                           action_space,
                                           **kwargs)
        # self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        # self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

    def _build_network(self) -> nn.Sequential:
        """Build network."""
        ff = mlp([self.observation_dim + self.action_dim] + self.hidden_layers + [1],
                 activation=nn.ReLU)
        return ff

    # def forward(self, state):
    #     """Forward pass in the actor network."""
    #     prob = self.fc1(state)
    #     prob = F.relu(prob)
    #
    #     prob = self.fc2(prob)
    #     prob = F.relu(prob)
    #
    #     mu = self.mu(prob)
    #     sigma = self.sigma(prob)
    #
    #     # Softplus activation function
    #     sigma = T.clamp(sigma, min=self.param_noise, max=1)
    #
    #     return mu, sigma
    #
    # def sample_normal(self, state, reparamterize=True):
    #     """Sample from a normal distribution."""
    #     mu, sigma = self.forward(state)
    #     probabilities = Normal(mu, sigma)
    #
    #     if reparamterize:  # Reparameterization trick: Sample with noise
    #         actions = probabilities.rsample()
    #     else:
    #         actions = probabilities.sample()
    #
    #     action = T.tanh(actions) * T.tensor(self.max_action).to(self.device)
    #     log_probs = probabilities.log_prob(actions)
    #     log_probs -= T.log(1 - action.pow(2) + self.param_noise)
    #     log_probs = log_probs.sum(1, keepdim=True)
    #
    #     return action, log_probs


class SACPolicy(nn.Module):
    """Policy for SAC algorithm."""

    def __init__(self,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
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

    def act(self, state: np.ndarray) -> np.ndarray:
        """Get action from the policy.

        args:
            state: Current state.

        returns:
            action: Action to take.

        """
        state = th.tensor(state, dtype=th.float32, device=self.actor.device)
        action, _ = self.actor.sample_normal(state, reparamterize=False)
        return action.cpu().detach().numpy()

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
