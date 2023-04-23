"""Module that defines the IDHP-SAC policy."""
from typing import Optional

import torch as th
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces

from agents import BasePolicy
from agents.idhp.idhp import IDHP
from agents.idhp.policy import Actor as IDHPActor
from agents.sac.policy import ActorNetwork as SACActor
from agents.sac.sac import SAC
from helpers.torch_helpers import freeze, mlp


class HybridActor(IDHPActor):
    """Class that implements the actor network for the IDHP-SAC agent."""

    def __init__(
        self,
        idhp_actor: IDHPActor,
        sac_actor: SACActor,
        *args,
        device: Optional[str] = None,
        **kwargs
    ):
        """Initialize the actor network."""
        super().__init__(
            *args,
            **kwargs,
            observation_space=idhp_actor.observation_space,
            action_space=idhp_actor.action_space,
            hidden_layers=idhp_actor.hidden_layers,
            lr_high=idhp_actor.lr_high,
            lr_low=idhp_actor.lr_low,
            lr_threshold=idhp_actor.lr_threshold,
            device=device,
        )
        freeze(sac_actor)
        self.sac = sac_actor
        self.sac_hidden = sac_actor.hidden_layers
        self.idhp_hidden = idhp_actor.hidden_layers
        self._setup_ff()
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        self.to(self.device)

    def _setup_ff(self):
        """Setup the feedforward network."""
        idhp_features = self.sac_hidden + [self.sac.mu.in_features]
        new_idhp = mlp(
            idhp_features, activation=nn.Tanh, output_activation=nn.Tanh, bias=False
        )
        self.ff = nn.Sequential()

        # Append an IDHP layer after a SAC layer
        for idx, layer in enumerate(self.sac.ff):
            self.ff.append(layer)
            if not isinstance(layer, nn.Linear):
                th.nn.init.eye_(
                    new_idhp[0].weight
                )  # Initialize the Linear layer as identity
                self.ff.append(new_idhp.pop(0))
                self.ff.append(new_idhp.pop(0))

    def forward(
        self, obs: th.Tensor, deterministic: bool = True, to_scale: bool = False
    ):
        output_idhp = self.ff(obs)
        action, _ = self.sac.output_layer(
            output_idhp, deterministic=deterministic, with_log_prob=False
        )
        return action


class SumActor(IDHPActor):
    """Class that implements the actor network for the IDHP-SAC agent that adds both networks."""

    def __init__(
        self,
        idhp_actor: IDHPActor,
        sac_actor: SACActor,
        *args,
        device: Optional[str] = None,
        **kwargs
    ):
        """Initialize the actor network."""
        super().__init__(
            *args,
            **kwargs,
            observation_space=idhp_actor.observation_space,
            action_space=idhp_actor.action_space,
            hidden_layers=idhp_actor.hidden_layers,
            lr_high=idhp_actor.lr_high,
            lr_low=idhp_actor.lr_low,
            lr_threshold=idhp_actor.lr_threshold,
            device=device,
        )
        freeze(sac_actor)
        self.sac = sac_actor
        self.ff = idhp_actor.ff
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        self.to(self.device)

    def forward(
        self, obs: th.Tensor, deterministic: bool = True, to_scale: bool = False
    ):
        output_idhp = self.ff(obs)
        output_sac, _ = self.sac(obs, deterministic=True, with_log_prob=False)
        output = output_idhp + output_sac
        return output


class SequentialActor(IDHPActor):
    """Class that implements the sequential version of the SAC-IDHP agent."""

    def __init__(
        self,
        idhp_actor: IDHPActor,
        sac_actor: SACActor,
        *args,
        device: Optional[str] = None,
        **kwargs
    ):
        """Initialize the actor network."""
        super().__init__(
            *args,
            **kwargs,
            observation_space=idhp_actor.observation_space,
            action_space=idhp_actor.action_space,
            hidden_layers=idhp_actor.hidden_layers,
            lr_high=idhp_actor.lr_high,
            lr_low=idhp_actor.lr_low,
            lr_threshold=idhp_actor.lr_threshold,
            device=device,
        )

        freeze(sac_actor)
        self.sac = sac_actor
        self.idhp = idhp_actor
        self.to(self.device)

    def setup_idhp(self):
        """Modify first layer of IDHP to make it have the sac output also as input."""
        self.idhp.ff[0] = nn.Linear(
            self.idhp.ff[0].in_features + self.sac.mu.in_features,
            self.idhp.ff[0].out_features,
        )

    def forward(self):
        """Forward pass."""
        pass


class IDHPSACPolicy(BasePolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Optional[str] = None,
    ):
        """Initialize the policy."""
        super().__init__(observation_space, action_space, device=device)

    def _setup_policy(self):
        """Setup the policy."""
        pass

    def transfer_learning(self, sac: SAC, idhp: IDHP):
        return HybridActor(idhp.policy.actor, sac.policy.actor, device=self.device)

    def predict(self, observation, deterministic: bool = True):
        """Predict the action."""
        pass
