"""Tests the IDHP agent."""
import pytest

from agents import IDHP
from envs import LTIEnv, CitationEnv, BaseEnv
from envs.rewards import get_reward
from envs.observations import get_observation
@pytest.mark.parametrize('env', [LTIEnv, CitationEnv])
class TestIDHP:
    """Tests the IDHP agent."""

    def test_init(self, env: BaseEnv):
        """Tests the IDHP agent."""
        agent = IDHP('default', env())
        assert agent is not None

    def test_setup_model(self, env: BaseEnv):
        """Tests the IDHP agent."""
        agent = IDHP('default', env())

        assert agent.model is not None
        assert agent.actor is not None
        assert agent.critic is not None

    def test_setup_env(self, env: BaseEnv):
        """Tests the IDHP agent."""
        agent = IDHP('default', env())
        assert agent.env is not None
        assert agent._env.env.get_reward == get_reward('sq_error')
        assert agent._env.env.get_obs == get_observation('states + ref')

    def test_learn(self, env: BaseEnv):
        """Tests the IDHP agent."""
        agent = IDHP('default', env())
        agent.learn(100)
        assert agent.num_timesteps == 100