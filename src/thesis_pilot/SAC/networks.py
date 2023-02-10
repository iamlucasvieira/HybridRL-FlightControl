"""Create networks for SAC algorithm."""
import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

from src.helpers.paths import Path
from src.helpers.misc import get_device

MODELS_PATH = Path.root / "envs"
TEMP_PATH = MODELS_PATH / "custom_sac"


class BaseNetwork(nn.Module):
    """Base network for Critic and Value networks."""

    def __init__(self, beta, input_dims, fc1_dims=256, fc2_dims=256, name='base', chkpt_dir=TEMP_PATH):
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
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = chkpt_dir / f'{name}_sac'

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.y = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = get_device()
        self.to(self.device)

    def save_checkpoint(self):
        """Save checkpoint."""
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """Load checkpoint."""
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(BaseNetwork):
    """Creates the critic neural network."""

    def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256, name='critic', chkpt_dir=TEMP_PATH):
        """Initialize critic network.

        args:
            beta: Learning rate.
            input_dims: Input dimensions.
            n_actions: Number of actions.
            fc1_dims: Number of neurons in the first layer.
            fc2_dims_: Number of neurons in the second layer.
        """
        # Input layer of critic's neural network in SAC uses state-action pairs
        input_dims = (input_dims[0] + n_actions,)
        super(CriticNetwork, self).__init__(beta, input_dims,
                                            fc1_dims=fc1_dims,
                                            fc2_dims=fc2_dims,
                                            name=name,
                                            chkpt_dir=chkpt_dir)

    def forward(self, state, action):
        """Forward pass of the critic's neural network."""
        # First layer
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value)

        # Second layer
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        # Output layer
        q = self.y(action_value)

        return q


class ValueNetwork(BaseNetwork):
    """Value function network SAC."""

    def __init__(self, beta, input_dims, fc1_dims=256, fc2_dims=256, name='value', chkpt_dir=TEMP_PATH):
        """Initialize value network.

        args:
            beta: Learning rate.
            input_dims: Input dimensions.
            fc1_dims: Number of neurons in the first layer.
            fc2_dims_: Number of neurons in the second layer.
            name: Name of the network.
            chkpt_dir: Checkpoint directory.
        """
        super(ValueNetwork, self).__init__(beta, input_dims,
                                           fc1_dims=fc1_dims,
                                           fc2_dims=fc2_dims,
                                           name=name,
                                           chkpt_dir=chkpt_dir)

    def forward(self, state):
        """Forward pass of the value network."""
        state_value = self.fc1(state)
        state_value = F.relu(state_value)

        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.y(state_value)

        return v


class ActorNetwork(BaseNetwork):
    """Actor network in SAC."""

    def __init__(self, alpha, input_dims, max_action, fc1_dims=256, fc2_dims=256, n_actions=2,
                 name='actor', chkpt_dir=TEMP_PATH):
        """Initialize actor network.

        args:
            alpha: Learning rate.
            input_dims: Input dimensions.
            fc1_dims: Number of neurons in the first layer.
            fc2_dims_: Number of neurons in the second layer.
            name: Name of the network.
            chkpt_dir: Checkpoint directory.
        """
        super(ActorNetwork, self).__init__(alpha, input_dims,
                                           fc1_dims=fc1_dims,
                                           fc2_dims=fc2_dims,
                                           name=name,
                                           chkpt_dir=chkpt_dir)
        self.max_action = max_action
        self.n_actions = n_actions
        self.param_noise = 1e-6

        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

    def forward(self, state):
        """Forward pass in the actor network."""
        prob = self.fc1(state)
        prob = F.relu(prob)

        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        # Softplus activation function
        sigma = T.clamp(sigma, min=self.param_noise, max=1)

        return mu, sigma

    def sample_normal(self, state, reparamterize=True):
        """Sample from a normal distribution."""
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparamterize:  # Reparameterization trick: Sample with noise
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = T.tanh(actions) * T.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1 - action.pow(2) + self.param_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs