"""Module that tests the DSAC agent."""
from typing import Type

import pytest

from agents import DSAC
from agents.buffer import Transition
from agents.dsac.policy import DSACPolicy
from envs import BaseEnv, CitationEnv, LTIEnv


@pytest.fixture
def transition(env: Type[BaseEnv]) -> Transition:
    """Returns a transition."""
    env = env()
    obs, _ = env.reset()
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    return Transition(obs=obs, action=action, reward=reward, obs_=next_obs, done=done)


@pytest.mark.parametrize("env", [LTIEnv, CitationEnv])
class TestDSAC:
    """Class that tests the DSAC objct."""

    def test_init(self, env: Type[BaseEnv]):
        """Tests the initialization of the DSAC object."""
        agent = DSAC(env())
        assert isinstance(agent, DSAC)
        assert isinstance(agent.policy, DSACPolicy)
        assert isinstance(agent.target_policy, DSACPolicy)

    def test_update(self, env: Type[BaseEnv], transition):
        """Tests the update method of the DSAC object."""
        env = env()
        agent = DSAC(env, buffer_size=100, batch_size=10)
        while not agent.replay_buffer.full():
            agent.replay_buffer.push(transition)

        n_updates = 5
        for _ in range(n_updates):
            agent.update()
        assert agent._n_updates == 5
