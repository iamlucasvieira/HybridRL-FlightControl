"""Module that tests the IDHP-SAC policy."""
import pytest
import torch as th

from agents.idhp.policy import Actor as IDHPActor
from agents.idhp_sac.policy import IDHPSACActor
from agents.sac.policy import ActorNetwork as SACActor
from envs import LTIEnv, CitationEnv


@pytest.fixture
def actor(env):
    """Fixture that returns an IDHP-SAC actor."""
    env = env()
    idhp_actor = IDHPActor(env.observation_space, env.action_space)
    sac_actor = SACActor(env.observation_space, env.action_space)
    actor = IDHPSACActor(idhp_actor, sac_actor)
    return actor


@pytest.mark.parametrize("env", [CitationEnv, LTIEnv])
class TestIDHPSACActor:
    """Class that thest the IDHP-SAC Actor."""

    def test_init(self, env, actor):
        """Method that tests the initialization of the actor."""
        assert actor is not None

    def test_forward(self, env, actor):
        """Method that tests the forward method."""
        obs = env().reset()
        actor.forward(th.as_tensor(obs))