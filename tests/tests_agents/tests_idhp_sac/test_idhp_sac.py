"""Tests for the IDHP-SAC agent."""

import pytest
import torch as th

from agents import IDHPSAC
from agents.idhp_sac.policy import IDHPSACActor
from envs import CitationEnv, LTIEnv
from envs.observations import get_observation
from envs.rewards import get_reward


@pytest.mark.parametrize("env", [CitationEnv, LTIEnv])
class TestIDHPSAC:
    """Test the IDHPSAC class."""

    def test_init(self, env):
        """Test the initialization of the IDHP-SAC agent."""
        env = env()
        agent = IDHPSAC("default", env)

        assert agent is not None

    def test_learn(self, env):
        """Test the learn method."""
        env = env()
        agent = IDHPSAC("default", env,
                        learning_starts=1,
                        buffer_size=1,
                        batch_size=1)

        agent.learn(sac_timesteps=3, idhp_timesteps=3)
        assert isinstance(agent.idhp.policy.actor, IDHPSACActor)
        assert agent.idhp.policy.actor == agent.idhp.actor
        assert agent.sac.num_timesteps == 3
        assert agent.sac._n_updates == 2
        assert agent.idhp.num_timesteps == 3

    def test_before_setup_idhp(self, env):
        """Tests that SAC and IDHP layers are not the same before setup."""
        env = env()
        agent = IDHPSAC("default", env,
                        learning_starts=1,
                        buffer_size=1,
                        batch_size=1)

        for idhp_layer, sac_layer in zip(agent.idhp.policy.actor.ff, agent.sac.policy.actor.ff):
            assert idhp_layer != sac_layer

    def test_setup_idhp(self, env):
        """Tests that SAC and IDHP layers are the same after setup."""
        env = env()
        agent = IDHPSAC("default", env,
                        learning_starts=1,
                        buffer_size=1,
                        batch_size=1)

        agent._setup_idhp()
        assert isinstance(agent.idhp.policy.actor, IDHPSACActor)

        for idhp_layer, sac_layer in zip(agent.idhp.policy.actor.sac.ff, agent.sac.policy.actor.ff):
            assert idhp_layer == sac_layer
            if isinstance(idhp_layer, th.nn.Linear):
                assert th.allclose(idhp_layer.weight, sac_layer.weight)
                assert th.allclose(idhp_layer.bias, sac_layer.bias)

    def test_frozen_idhp_layers(self, env):
        """Tests that the first IDHP layers are frozen and weights are the same."""
        env = env()
        agent = IDHPSAC("default", env,
                        learning_starts=1,
                        buffer_size=1,
                        batch_size=1)

        agent.learn(sac_timesteps=3, idhp_timesteps=3)

        for idhp_layer, sac_layer in zip(agent.idhp.policy.actor.sac.ff, agent.sac.policy.actor.ff):
            if isinstance(idhp_layer, th.nn.Linear):
                assert th.allclose(idhp_layer.weight, sac_layer.weight)
                assert th.allclose(idhp_layer.bias, sac_layer.bias)

    def test_envs_independent(self, env):
        """Tests that the envs from SAC and IDHP are different."""
        env = env()
        agent = IDHPSAC("default", env,
                        learning_starts=1,
                        buffer_size=1,
                        batch_size=1)

        assert agent.sac._env != agent.idhp._env
        assert agent.sac._env.episode_steps == agent.idhp._env.episode_steps
        agent.idhp._env.episode_steps = 0
        assert agent.sac._env.episode_steps != agent.idhp._env.episode_steps

    def test_envs_are_idhp_like(self, env):
        """Tests if the created environment are according to IDHP requirements."""
        env = env()
        agent = IDHPSAC("default", env,
                        learning_starts=1,
                        buffer_size=1,
                        batch_size=1)

        reward_function = 'sq_error'
        observation_function = 'states + ref'

        assert agent.sac._env.get_reward == get_reward(reward_function)
        assert agent.idhp._env.get_reward == get_reward(reward_function)

        assert agent.sac._env.get_obs == get_observation(observation_function)
        assert agent.idhp._env.get_obs == get_observation(observation_function)
