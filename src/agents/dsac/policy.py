"""Module that defines the DSAC policy."""

from typing import Optional

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn

from agents import BasePolicy
from agents.sac.policy import ActorNetwork
from helpers.torch_helpers import (
    BaseNetwork,
    check_dimensions,
    check_shape,
    get_device,
    mlp,
)


class CriticNetwork(BaseNetwork):
    """Defines the critic network of the DSAC policy."""

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        device: Optional[str] = None,
        hidden_layers: Optional[list] = None,
        embedding_dim: int = 64,
        layer_norm: bool = True,
        learning_rate: float = 3e-4,
        **kwargs
    ):
        """Initialize critic network."""
        if hidden_layers is None:
            hidden_layers = [256, 256]
        device = get_device() if device is None else device

        super().__init__(
            observation_space,
            action_space,
            hidden_layers=hidden_layers,
            device=device,
            learning_rate=learning_rate,
            **kwargs,
            build_network=False
        )

        self.iqn = IQN(
            observation_space,
            action_space,
            device=device,
            hidden_layers=hidden_layers,
            embedding_dim=embedding_dim,
            layer_norm=layer_norm,
            learning_rate=learning_rate,
        )

        self.to(self.device)
        self._build_optimizer()

    def forward(
        self, state: th.Tensor, action: th.Tensor, quantile: th.Tensor
    ) -> th.Tensor:
        """Forward pass of the critic network.

        Args:
            state: State tensor of shape (B, D)
            action: Action tensor of shape (B, A)
            quantile: Quantile tensor of shape (B, Q, 1)

        Dimensions guide:
        - B: Batch size
        - Q: Number of quantiles
        - S: State dimension
        - A: Action dimension
        - H: Last Hidden layer dimension
        - E: Embedding dimension
        """
        quantile_per_action = self.iqn(state, action, quantile)  # (B, Q, A)
        # Remove action dimension
        quantile = th.squeeze(quantile_per_action, dim=-1)  # (B, Q)
        return quantile


class IQN(BaseNetwork):
    """Network for the quantile regression."""

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        device: Optional[str] = None,
        hidden_layers: Optional[list] = None,
        embedding_dim: int = 64,
        layer_norm: bool = True,
        learning_rate: float = 3e-4,
        **kwargs
    ):
        """Initialize critic network."""
        if hidden_layers is None:
            hidden_layers = [256, 256]
        elif len(hidden_layers) < 2:
            raise ValueError("Hidden layers must have at least two layers")

        device = get_device() if device is None else device

        super().__init__(
            observation_space,
            action_space,
            hidden_layers=hidden_layers,
            device=device,
            build_network=False,
            learning_rate=learning_rate,
            **kwargs
        )

        # Define the state feature layer (psi)
        psi_layers = []
        last_dim = self.observation_dim + self.action_dim
        for hidden_layer_dim in self.hidden_layers[:-1]:
            psi_layers.append(nn.Linear(last_dim, hidden_layer_dim))
            if layer_norm:
                psi_layers.append(nn.LayerNorm(hidden_layer_dim))
            psi_layers.append(nn.ReLU())
            last_dim = hidden_layer_dim

        self.psi = nn.Sequential(*psi_layers)

        # Define the embedding layer (phi)
        self.phi = nn.Sequential(
            nn.Linear(embedding_dim, self.hidden_layers[-2]),
            nn.LayerNorm(self.hidden_layers[-2]) if layer_norm else nn.Identity(),
            nn.Sigmoid(),
        )

        # Define the merge layer (rho)
        self.merge = nn.Sequential(
            nn.Linear(self.hidden_layers[-2], self.hidden_layers[-1]),
            nn.LayerNorm(self.hidden_layers[-1]) if layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(self.hidden_layers[-1], 1),
        )

        self.constant_vector = (
            th.from_numpy(np.arange(1, 1 + embedding_dim)).float().to(self.device)
        )
        self.embedding_dim = embedding_dim

        self.to(self.device)
        self._build_optimizer()

    def forward(
        self, state: th.Tensor, action: th.Tensor, quantile: th.Tensor
    ) -> th.Tensor:
        """Build network.

        Args:
            state: State tensor of shape (B, D)
            action: Action tensor of shape (B, A)
            quantile: Quantile tensor of shape (B, Q, 1)

        Dimensions guide:
        - B: Batch size
        - Q: Number of quantiles
        - S: State dimension
        - A: Action dimension
        - H: Last Hidden layer dimension
        - E: Embedding dimension
        """
        # Ensure correct dimensions
        check_dimensions(state, 2, name="State vector")
        check_dimensions(action, 2, name="Action vector")
        check_dimensions(quantile, 3, name="Quantile vector")

        # Define sizes
        _B = state.shape[0]
        _Q = quantile.shape[1]
        _S = state.shape[1]
        _A = action.shape[1]
        _H = self.hidden_layers[-1]
        _E = self.embedding_dim

        # Move inputs to device
        state = state.to(self.device)
        action = action.to(self.device)
        quantile = quantile.to(self.device)

        # Compute psi output
        state_action = th.cat([state, action], dim=1)  # (B, S + A)
        check_shape(state_action, (_B, _S + _A), "State-action vector")
        psi = self.psi(state_action)  # (B, H)
        check_shape(psi, (_B, _H), "Psi output")

        # Compute phi output
        cos_tau = th.cos(quantile * self.constant_vector * th.pi)  # (B, Q, E)
        check_shape(cos_tau, (_B, _Q, _E), "Cosine of quantile")
        phi = self.phi(cos_tau)  # (B, Q, H)
        check_shape(phi, (_B, _Q, _H), "Phi output")

        # Compute merge of psi and phi
        psi_phi_mult = th.mul(phi, psi.view(_B, 1, _H))  # (B, Q, H)
        check_shape(psi_phi_mult, (_B, _Q, _H), "Psi-phi multiplication")
        output = self.merge(psi_phi_mult)  # (B, Q, 1)
        check_shape(output, (_B, _Q, 1), "Merge output")

        return output


class DSACPolicy(BasePolicy):
    """DSAC policy."""

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        device: Optional[str] = None,
        hidden_layers: Optional[list] = None,
        embedding_dim: int = 64,
        layer_norm: bool = True,
        learning_rate: float = 3e-4,
    ):
        """Initialize DSAC policy.

        Args:
            observation_space: Observation space
            action_space: Action space
            device: Device to use for tensor operations
            hidden_layers: Number of units per layer
            embedding_dim: Embedding dimension
            layer_norm: Whether to use layer normalization
        """
        if device is None:
            device = get_device()

        self.device = device
        self.hidden_layers = hidden_layers
        self.embedding_dim = embedding_dim
        self.layer_norm = layer_norm
        self.learning_rate = learning_rate

        super().__init__(observation_space, action_space)
        self.to(device)

    def _setup_policy(self):
        self.z1 = CriticNetwork(
            self.observation_space,
            self.action_space,
            hidden_layers=self.hidden_layers,
            embedding_dim=self.embedding_dim,
            layer_norm=self.layer_norm,
            learning_rate=self.learning_rate,
            device=self.device,
        )
        self.z2 = CriticNetwork(
            self.observation_space,
            self.action_space,
            hidden_layers=self.hidden_layers,
            embedding_dim=self.embedding_dim,
            layer_norm=self.layer_norm,
            learning_rate=self.learning_rate,
            device=self.device,
        )
        self.actor = ActorNetwork(
            self.observation_space,
            self.action_space,
            learning_rate=self.learning_rate,
        )

    def predict(
        self, observation: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        """Predict action for a given observation."""
        observation = th.as_tensor(observation).float().to(self.device)
        action, _ = self.actor(
            observation, deterministic=deterministic, with_log_prob=False
        )
        return action.detach().cpu().numpy()
