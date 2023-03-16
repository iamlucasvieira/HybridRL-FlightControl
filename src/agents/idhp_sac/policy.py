"""Module that defienes the IDHP-SAC policy."""

import torch as th
import torch.nn as nn

from agents.idhp.policy import Actor as IDHPActor
from agents.sac.policy import ActorNetwork as SACActor
from helpers.torch_helpers import freeze, mlp


class IDHPSACActor(IDHPActor):
    """Class that implements the actor network for the IDHP-SAC agent."""

    def __init__(self, idhp_actor: IDHPActor, sac_actor: SACActor, *args, **kwargs):
        """Initialize the actor network."""
        super(IDHPSACActor, self).__init__(*args, **kwargs,
                                           observation_space=idhp_actor.observation_space,
                                           action_space=idhp_actor.action_space,
                                           hidden_layers=idhp_actor.hidden_layers,
                                           learning_rate=idhp_actor.learning_rate)
        freeze(sac_actor)
        self.sac = sac_actor
        self.sac_hidden = sac_actor.hidden_layers
        self.idhp_hidden = idhp_actor.hidden_layers
        self._setup_ff()

    def _setup_ff(self):
        """Setup the feedforward network."""
        idhp_features = [self.sac_hidden[-1]] + self.idhp_hidden + [self.sac.mu.in_features]
        new_idhp = mlp(idhp_features, activation=nn.ReLU, output_activation=nn.Identity, bias=False)
        # for layer in new_idhp:
        #     if isinstance(layer, nn.Linear):
        #         nn.init.ones_(layer.weight)
        self.ff = new_idhp

    def forward(self, obs: th.Tensor, deterministic: bool = True, to_scale: bool = False):
        output_sac = self.sac.ff(obs)
        output_idhp = self.ff(output_sac)
        action, _ = self.sac.output_layer(output_idhp, deterministic=deterministic, with_log_prob=False)
        return action


class IDHPSACPolicy(nn.Module):

    def __init__(self):
        pass
