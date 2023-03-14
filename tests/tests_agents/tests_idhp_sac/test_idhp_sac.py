"""Tests for the IDHP-SAC agent."""
from copy import deepcopy

import pytest
import torch as th

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

    def test_learn(self, env):
        """Test the learn method."""
        env = env()
        agent = IDHPSAC("default", env,
                        learning_starts=1,
                        buffer_size=1,
                        batch_size=1)

        agent.learn(3)
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

        for idhp_layer, sac_layer in zip(agent.idhp.policy.actor.ff, agent.sac.policy.actor.ff):
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

        agent._setup_idhp()
        layers_before_learning = deepcopy(agent.sac.policy.actor.ff)
        agent.learn(3)

        for idhp_layer, sac_layer in zip(agent.idhp.policy.actor.ff, layers_before_learning):
            if isinstance(idhp_layer, th.nn.Linear):
                assert th.allclose(idhp_layer.weight, sac_layer.weight)
                assert th.allclose(idhp_layer.bias, sac_layer.bias)
