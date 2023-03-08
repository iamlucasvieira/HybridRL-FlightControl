"""Module that tests the implementation of the SAC algorithm."""

import pytest

from agents.sac.sac import SAC
from agents.sac.policy import SACPolicy
from agents.sac.buffer import ReplayBuffer
from envs.citation.citation_env import CitationEnv


@pytest.fixture
def env():
    """Create a CitationEnv instance."""
    return CitationEnv()


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
        assert isinstance(sac.critic_policy, SACPolicy)
        assert sac.critic_policy is not sac.policy

    def test_replay_buffer(self, env):
        """Test that SAC replay buffer is correctly initialized."""
        sac = SAC(env)
        assert isinstance(sac.replay_buffer, ReplayBuffer)
