"""Module that defines the IDHP-SAC policy."""
from typing import Optional

import torch as th
import torch.nn as nn
from gymnasium import spaces

from agents import BasePolicy
from agents.sac.sac import SAC
from agents.idhp.idhp import IDHP
from agents.idhp.policy import Actor as IDHPActor
from agents.sac.policy import ActorNetwork as SACActor
from helpers.torch_helpers import freeze, mlp


class IDHPSACActor(IDHPActor):
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
            learning_rate=idhp_actor.learning_rate,
            device=device,
        )
        freeze(sac_actor)
        self.sac = sac_actor
        self.sac_hidden = sac_actor.hidden_layers
        self.idhp_hidden = idhp_actor.hidden_layers
        self._setup_ff()
        self.to(self.device)

    def _setup_ff(self):
        """Setup the feedforward network."""
        idhp_features = self.sac_hidden + [self.sac.mu.in_features]
        new_idhp = mlp(
            idhp_features, activation=nn.ReLU, output_activation=nn.ReLU, bias=False
        )
        self.ff = nn.Sequential()

        # Append an IDHP layer after a SAC layer
        for idx, layer in enumerate(self.sac.ff):
            self.ff.append(layer)
            if not isinstance(layer, nn.Linear):
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
        return IDHPSACActor(idhp.policy.actor, sac.policy.actor, device=self.device)

    def predict(self, observation, deterministic: bool = True):
        """Predict the action."""
        pass
