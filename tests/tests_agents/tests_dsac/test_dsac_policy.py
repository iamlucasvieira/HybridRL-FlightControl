"""Module that tests the DSAC policy."""

import pytest
import torch as th
from torch import nn

from agents.dsac.policy import IQN, CriticNetwork, DSACPolicy
from agents.sac.policy import ActorNetwork
from envs import CitationEnv, LTIEnv


hidden_layers_params = [[156, 256], [412, 412, 512], [1021, 1022, 1023, 1024]]


@pytest.mark.parametrize("env", [LTIEnv, CitationEnv])
class TestIQN:
    """Tests the IQN network."""

    def test_iqn_init(self, env):
        """Test IQN network."""
        env = env()
        iqn = IQN(env.observation_space, env.action_space)
        assert isinstance(iqn, IQN)

    @pytest.mark.parametrize("hidden_layers", hidden_layers_params)
    def test_iqn_psi(self, env, hidden_layers):
        """Test the psi layer."""
        env = env()
        observation_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        # With layer norm:
        iqn = IQN(env.observation_space, env.action_space, hidden_layers=hidden_layers)

        psi = iqn.psi
        assert psi[0].in_features == observation_dim + action_dim
        assert psi[-3].out_features == hidden_layers[-2]
        assert len(psi) == (len(hidden_layers) - 1) * 3

        # Without layer norm:
        iqn_no_ln = IQN(
            env.observation_space,
            env.action_space,
            hidden_layers=hidden_layers,
            layer_norm=False,
        )
        psi_no_ln = iqn_no_ln.psi
        assert psi_no_ln[0].in_features == observation_dim + action_dim
        assert psi_no_ln[-2].out_features == hidden_layers[-2]
        assert len(psi_no_ln) == (len(hidden_layers) - 1) * 2

    @pytest.mark.parametrize("hidden_layers", hidden_layers_params)
    def test_iqn_phi(self, env, hidden_layers):
        """Test the phi layer."""
        env = env()
        embedding_dim = hidden_layers[-1]

        # With layer norm:
        iqn = IQN(
            env.observation_space,
            env.action_space,
            hidden_layers=hidden_layers,
            embedding_dim=embedding_dim,
        )

        phi = iqn.phi
        assert phi[0].in_features == embedding_dim
        assert phi[-3].out_features == hidden_layers[-2]
        assert len(phi) == 3

        # Without layer norm:
        iqn_no_ln = IQN(
            env.observation_space,
            env.action_space,
            hidden_layers=hidden_layers,
            embedding_dim=embedding_dim,
            layer_norm=False,
        )
        phi_no_ln = iqn_no_ln.phi
        assert phi_no_ln[0].in_features == embedding_dim
        assert phi_no_ln[-3].out_features == hidden_layers[-2]
        assert isinstance(phi_no_ln[-2], nn.Identity)
        assert len(phi_no_ln) == 3

    @pytest.mark.parametrize("hidden_layers", hidden_layers_params)
    def test_iqn_merge(self, env, hidden_layers):
        """Test the merge layer."""
        env = env()

        # With layer norm:
        iqn = IQN(
            env.observation_space,
            env.action_space,
            hidden_layers=hidden_layers,
        )

        merge = iqn.merge
        assert merge[0].in_features == hidden_layers[-2]
        assert merge[0].out_features == hidden_layers[-1]
        assert merge[-1].out_features == 1
        assert len(merge) == 4

        # Without layer norm:
        iqn_no_ln = IQN(
            env.observation_space,
            env.action_space,
            hidden_layers=hidden_layers,
            layer_norm=False,
        )
        merge_no_ln = iqn_no_ln.merge
        assert merge_no_ln[0].in_features == hidden_layers[-2]
        assert isinstance(merge_no_ln[1], nn.Identity)
        assert merge_no_ln[0].out_features == hidden_layers[-1]
        assert merge_no_ln[-1].out_features == 1
        assert len(merge_no_ln) == 4

    def test_forward_1d_input(self, env):
        """Test the forward pass."""
        env = env()
        iqn = IQN(env.observation_space, env.action_space)
        obs, _ = env.reset()
        obs = th.as_tensor(obs)
        action = th.as_tensor(env.action_space.sample())

        # Assert that raises state shape error
        with pytest.raises(ValueError) as exc_info:
            iqn(obs, action.view(1, -1), obs.view(1, -1, 1))
        assert "State" in str(exc_info.value)

        # Assert that raises action shape error
        with pytest.raises(ValueError) as exc_info:
            iqn(obs.view(1, -1), action, obs.view(1, -1, 1))
        assert "Action" in str(exc_info.value)

        # Assert that raises quantile shape error
        with pytest.raises(ValueError) as exc_info:
            iqn(obs.view(1, -1), action.view(1, -1), obs)
        assert "Quantile" in str(exc_info.value)

    def test_forward_state_action_different_batch_size(self, env):
        """Test the forward pass raises error when state and action have different batch size."""
        env = env()
        iqn = IQN(env.observation_space, env.action_space)
        obs, _ = env.reset()
        obs = th.as_tensor(obs)
        action = th.as_tensor([env.action_space.sample() for _ in range(2)])

        # Assert that raises state shape error
        with pytest.raises(RuntimeError) as exc_info:
            iqn(obs.view(1, -1), action, obs.view(1, -1, 1))
        assert "Sizes of tensors must match except in dimension 1" in str(
            exc_info.value
        )

    def test_forward_2d_input(self, env):
        """Test the forward pass with correct shapes."""
        env = env()
        iqn = IQN(env.observation_space, env.action_space)
        obs, _ = env.reset()
        obs = th.as_tensor(obs)
        action = th.as_tensor(env.action_space.sample())
        quantile = th.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]).view(1, -1, 1)
        output = iqn(obs.view(1, -1), action.view(1, -1), quantile)
        assert output.shape == (1, quantile.shape[1], 1)

    @pytest.mark.parametrize("batch_size", [1, 3, 9])
    def test_forward_multiple_batch(self, env, batch_size):
        """Test the forward pass with multiple batch sizes."""
        env = env()
        iqn = IQN(env.observation_space, env.action_space)

        obs, _ = env.reset()
        obs = th.as_tensor([obs for _ in range(batch_size)])

        action = th.as_tensor([env.action_space.sample() for _ in range(batch_size)])

        quantile = th.linspace(0, 1, 9).view(batch_size, -1, 1)

        output = iqn(obs, action, quantile)
        assert output.shape == (batch_size, quantile.shape[1], 1)

    def test_1d_hidden_layer(self, env):
        """Test the forward pass with 1d hidden layer."""
        env = env()

        with pytest.raises(ValueError) as exc_info:
            IQN(env.observation_space, env.action_space, hidden_layers=[1])
        assert "Hidden layers" in str(exc_info.value)

    def test_optimizer_exists(self, env):
        """Test the optimizer exists."""
        env = env()
        critic = IQN(env.observation_space, env.action_space)
        assert critic.optimizer is not None


@pytest.mark.parametrize("env", [LTIEnv, CitationEnv])
class TestCritic:
    """Test the Critic class."""

    def test_init(self, env):
        """Test the initialization of the Critic class."""
        env = env()
        critic = CriticNetwork(env.observation_space, env.action_space)
        assert isinstance(critic.iqn, IQN)

    @pytest.mark.parametrize("batch_size", [1, 3, 9])
    def test_forward(self, env, batch_size):
        """Test the forward pass of the Critic class."""
        env = env()

        obs, _ = env.reset()
        obs = th.as_tensor([obs for _ in range(batch_size)])

        action = th.as_tensor([env.action_space.sample() for _ in range(batch_size)])

        quantile = th.linspace(0, 1, 9).view(batch_size, -1, 1)

        critic = CriticNetwork(env.observation_space, env.action_space)
        output = critic(obs, action, quantile)
        assert output.shape == (batch_size, quantile.shape[1])

    def test_optimizer_exists(self, env):
        """Test the optimizer exists."""
        env = env()
        critic = CriticNetwork(env.observation_space, env.action_space)
        assert critic.optimizer is not None


@pytest.mark.parametrize("env", [LTIEnv, CitationEnv])
class TestDSACPolicy:
    """Tests the DSAC policy."""

    def test_init(self, env):
        """Test the initilisation of the policy."""
        env = env()
        policy = DSACPolicy(env.observation_space, env.action_space)
        assert isinstance(policy.actor, ActorNetwork)
        assert isinstance(policy.z1, CriticNetwork)
        assert isinstance(policy.z2, CriticNetwork)

    def test_predict(self, env):
        """Test the predict method of the policy."""
        env = env()
        policy = DSACPolicy(env.observation_space, env.action_space)
        obs, _ = env.reset()
        action = policy.predict(obs)
        assert action.shape == env.action_space.shape
