"""Tests the IDHP agent."""
import pytest

from agents import IDHP
from envs import BaseEnv
from envs import CitationEnv
from envs import LTIEnv
from envs.observations import get_observation
from envs.rewards import get_reward


@pytest.mark.parametrize("env", [LTIEnv, CitationEnv])
class TestIDHP:
    """Tests the IDHP agent."""

    def test_init(self, env: BaseEnv):
        """Tests the IDHP agent."""
        agent = IDHP(env())
        assert agent is not None

    def test_setup_model(self, env: BaseEnv):
        """Tests the IDHP agent."""
        agent = IDHP(env())

        assert agent.model is not None
        assert agent.actor is not None
        assert agent.critic is not None

    def test_setup_env(self, env: BaseEnv):
        """Tests the IDHP agent."""
        agent = IDHP(env())
        assert agent.env is not None
        assert agent.env.get_reward == get_reward("sq_error")
        assert agent.env.get_obs == get_observation("states + ref")

    def test_learn(self, env: BaseEnv):
        """Tests the IDHP agent."""
        agent = IDHP(env())
        agent.learn(100)
        assert agent.num_steps == 100

    def test_policy_kwargs(self, env: BaseEnv):
        """Tests the IDHP agent with kwargs for policy."""
        agent = IDHP(
            env(),
            actor_kwargs={"hidden_layers": [10, 15, 20]},
            critic_kwargs={"hidden_layers": [15]},
        )
        assert agent.actor.num_hidden_layers == 3
        assert agent.critic.num_hidden_layers == 1
