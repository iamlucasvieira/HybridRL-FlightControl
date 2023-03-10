"""Module that tests the implementation of the SAC algorithm."""

import pytest

from agents.sac.sac import SAC
from agents.sac.policy import SACPolicy
from agents.sac.buffer import ReplayBuffer, Transition
from envs.citation.citation_env import CitationEnv
from envs.lti_citation.lti_env import LTIEnv
import torch as th


@pytest.fixture
def transition(env):
    """Create a Transition instance."""
    obs = env.reset()
    return Transition(obs=th.from_numpy(obs),
                      action=th.from_numpy(env.action_space.sample()),
                      reward=0,
                      obs_=th.from_numpy(obs),
                      done=False)


@pytest.mark.parametrize('env', [CitationEnv(), LTIEnv()], ids=['CitationEnv', 'LTIEnv'])
class TestSAC:
    """Test SAC class."""

    def test_init(self, env):
        """Test that SAC is correctly initialized."""
        sac = SAC(env)
        assert sac is not None

    def test_policy(self, env):
        """Test that SAC policy is correctly initialized."""
        sac = SAC(env)
        assert isinstance(sac.policy, SACPolicy)

    def test_target_policy(self, env):
        """Test that SAC target policy is correctly initialized and that it is a different object than policy."""
        sac = SAC(env)
        assert isinstance(sac.target_policy, SACPolicy)
        assert sac.target_policy is not sac.policy

    def test_replay_buffer(self, env):
        """Test that SAC replay buffer is correctly initialized."""
        sac = SAC(env)
        assert isinstance(sac.replay_buffer, ReplayBuffer)

    def test_get_critic_loss(self, env, transition):
        """Test that SAC critic loss is correctly computed."""
        sac = SAC(env)
        critic_loss = sac.get_critic_loss(transition)
        assert isinstance(critic_loss, th.Tensor)

    def test_get_actor_loss(self, env, transition):
        """Test that SAC actor loss is correctly computed."""
        sac = SAC(env)
        actor_loss = sac.get_actor_loss(transition)
        assert isinstance(actor_loss, th.Tensor)
