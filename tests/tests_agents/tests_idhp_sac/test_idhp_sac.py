"""Tests for the IDHP-SAC agent."""

import pytest
import torch as th

from agents import IDHPSAC
from agents.idhp.policy import Actor as IDHPActor
from agents.idhp_sac.policy import HybridActor
from envs import CitationEnv, LTIEnv
from envs.observations import get_observation
from envs.rewards import get_reward


@pytest.mark.parametrize("env", [CitationEnv, LTIEnv])
class TestIDHPSAC:
    """Test the IDHPSAC class."""

    def test_init(self, env):
        """Test the initialization of the IDHP-SAC agent."""
        env = env()
        agent = IDHPSAC(env)
        assert agent is not None

    def test_learn(self, env):
        """Test the learn method."""
        env = env()
        agent = IDHPSAC(
            env, sac_kwargs=dict(learning_starts=1, buffer_size=1, batch_size=1)
        )

        agent.learn(sac_steps=3, idhp_steps=3)
        assert isinstance(agent.idhp.policy.actor, HybridActor)
        assert agent.idhp.policy.actor == agent.idhp.actor
        assert agent.sac.num_steps == 3
        assert agent.sac._n_updates == 2
        assert agent.idhp.num_steps == 3

    def test_idhp_policy_is_default_before_run(self, env):
        """Tests that IDHP actor is not modified before learn."""
        env = env()
        agent = IDHPSAC(
            env, sac_kwargs=dict(learning_starts=1, buffer_size=1, batch_size=1)
        )

        assert isinstance(agent.idhp.policy.actor, IDHPActor)

    def test_idhp_layers_after_learn(self, env):
        """Tests that the first IDHP layers are frozen and weights are the same."""
        env = env()
        agent = IDHPSAC(
            env, sac_kwargs=dict(learning_starts=1, buffer_size=1, batch_size=1)
        )

        agent.learn(sac_steps=3, idhp_steps=3)

        assert isinstance(agent.idhp.policy.actor, HybridActor)
        for idhp_layer, sac_layer in zip(
            agent.idhp.policy.actor.sac.ff, agent.sac.policy.actor.ff
        ):
            if isinstance(idhp_layer, th.nn.Linear):
                assert idhp_layer.weight.requires_grad is False
                assert th.allclose(idhp_layer.weight, sac_layer.weight)
                assert th.allclose(idhp_layer.bias, sac_layer.bias)

    def test_envs_independent(self, env):
        """Tests that the envs from SAC and IDHP are different."""
        env = env()
        agent = IDHPSAC(
            env, sac_kwargs=dict(learning_starts=1, buffer_size=1, batch_size=1)
        )

        assert agent.sac.env != agent.idhp.env
        assert agent.sac.env.episode_steps == agent.idhp.env.episode_steps
        agent.idhp.env.episode_steps = 0
        assert agent.sac.env.episode_steps != agent.idhp.env.episode_steps

    def test_envs_are_idhp_like(self, env):
        """Tests if the created environment are according to IDHP requirements."""
        env = env()
        agent = IDHPSAC(
            env, sac_kwargs=dict(learning_starts=1, buffer_size=1, batch_size=1)
        )

        reward_function = "sq_error"
        observation_function = "states + ref"

        assert agent.sac.env.get_reward == get_reward(reward_function)
        assert agent.idhp.env.get_reward == get_reward(reward_function)

        assert agent.sac.env.get_obs == get_observation(observation_function)
        assert agent.idhp.env.get_obs == get_observation(observation_function)
