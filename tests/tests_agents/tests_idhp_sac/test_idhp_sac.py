"""Tests for the IDHP-SAC agent."""
import pytest

from agents import IDHPSAC
from envs import CitationEnv, LTIEnv

@pytest.mark.parametrize("env", [CitationEnv, LTIEnv])
class TestIDHPSAC:
    """Test the IDHPSAC class."""
    def test_init(self, env):
        """Test the initialization of the IDHP-SAC agent."""
        env = env()
        agent = IDHPSAC("default", env)

        assert agent is not None

    def test_learn_sac(self, env):
        """Test the learn method of the IDHP-SAC agent for the sac."""
        env = env()
        agent = IDHPSAC("default", env,
                        learning_starts=1,
                        buffer_size=1,
                        batch_size=1)

        agent.learn(3)
        assert agent.sac.num_timesteps == 3
        assert agent.sac._n_updates == 2
