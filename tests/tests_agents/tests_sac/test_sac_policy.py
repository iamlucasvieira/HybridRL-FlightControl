import pytest
import torch as th
from torch import nn

from agents.sac.policy import CriticNetwork, ActorNetwork, SACPolicy
from envs.citation.citation_env import CitationEnv


@pytest.fixture
def env():
    """Create a CitationEnv instance."""
    return CitationEnv()


@pytest.fixture
def critic_kwargs():
    """Create a dictionary with critic network parameters."""
    return {
        "hidden_layers": [256, 256],
        "learning_rate": 0.001,
    }


class TestCriticNetwork:
    """Test CriticNetwork class."""

    @pytest.mark.parametrize("hidden_layers",
                             [[256, 257, 258, 259],
                              [256, 256],
                              [1, 2, 3],
                              [256]])
    def test_structure(self, env, critic_kwargs, hidden_layers):
        """Test if network creates correct layers."""
        observation_space = env.observation_space
        action_space = env.action_space

        critic_kwargs["hidden_layers"] = hidden_layers

        net = CriticNetwork(observation_space, action_space, **critic_kwargs)

        input_size = net.observation_dim + net.action_dim
        output_size = 1

        all_layers = [input_size] + hidden_layers + [output_size]

        for idx, layer in enumerate(net.ff):
            if idx % 2 == 0:
                assert isinstance(layer, nn.Linear)
                assert layer.in_features == all_layers[idx // 2]
                assert layer.out_features == all_layers[idx // 2 + 1]
            elif idx == len(net.ff) - 1:
                assert isinstance(layer, nn.Identity)
            else:
                assert isinstance(layer, nn.ReLU)

    def test_forward(self, env, critic_kwargs):
        """Test if forward pass returns correct shape."""
        observation_space = env.observation_space
        action_space = env.action_space

        net = CriticNetwork(observation_space, action_space, **critic_kwargs)

        state = env.reset()
        action = env.action_space.sample()

        output = net(th.tensor(state), th.tensor(action))

        assert output.shape == ()


@pytest.fixture
def actor_kwargs():
    """Create a dictionary with actor network parameters."""
    return {
        "hidden_layers": [256, 256],
        "learning_rate": 0.001,
    }


class TestActorNetwork:
    """Test ActorNetwork class."""

    @pytest.mark.parametrize("hidden_layers",
                             [[256, 257, 258, 259],
                              [256, 256],
                              [1, 2, 3],
                              [256]])
    def test_structure(self, env, actor_kwargs, hidden_layers):
        """Test if network creates correct layers."""
        observation_space = env.observation_space
        action_space = env.action_space

        actor_kwargs["hidden_layers"] = hidden_layers

        net = ActorNetwork(observation_space, action_space, **actor_kwargs)

        input_size = net.observation_dim

        all_layers = [input_size] + hidden_layers

        # Assert hidden layers are correct
        for idx, layer in enumerate(net.ff):
            if idx % 2 == 0:
                assert isinstance(layer, nn.Linear)
                assert layer.in_features == all_layers[idx // 2]
                assert layer.out_features == all_layers[idx // 2 + 1]
            else:
                assert isinstance(layer, nn.ReLU)

        # Assert output layers are correct
        assert isinstance(net.mu, nn.Linear)
        assert net.mu.in_features == all_layers[-1]
        assert net.mu.out_features == net.action_dim

        assert isinstance(net.log_sigma, nn.Linear)
        assert net.log_sigma.in_features == all_layers[-1]

    def test_forward(self, env, actor_kwargs):
        """Test if forward pass returns correct shape."""
        observation_space = env.observation_space
        action_space = env.action_space

        net = ActorNetwork(observation_space, action_space, **actor_kwargs)

        state = env.reset()

        action, log_prob = net(th.tensor(state))

        assert action.shape == action_space.shape
        assert log_prob is not None

    def test_forward_no_log_prob(self, env, actor_kwargs):
        """Test if forward pass returns correct shape."""
        observation_space = env.observation_space
        action_space = env.action_space

        net = ActorNetwork(observation_space, action_space, **actor_kwargs)

        state = env.reset()

        action, log_prob = net(th.tensor(state), with_log_prob=False)

        assert action.shape == action_space.shape
        assert log_prob is None

    def test_forward_no_reparam(self, env, actor_kwargs):
        """Test if forward pass returns correct shape without reparam."""
        observation_space = env.observation_space
        action_space = env.action_space

        net = ActorNetwork(observation_space, action_space, **actor_kwargs)

        state = env.reset()

        action, log_prob = net(th.tensor(state), deterministic=True)

        assert action.shape == action_space.shape
        assert log_prob is not None


class TestSACPolicy:
    """Test SACPolicy class."""

    def test_init(self, env):
        """Test if policy is correctly initialized."""
        observation_space = env.observation_space
        action_space = env.action_space

        policy = SACPolicy(observation_space, action_space)

        assert policy is not None
        assert isinstance(policy.actor, ActorNetwork)
        assert isinstance(policy.critic_1, CriticNetwork)
        assert isinstance(policy.critic_2, CriticNetwork)

    def test_get_action(self, env):
        """Test if get_action returns correct shapes."""
        observation_space = env.observation_space
        action_space = env.action_space

        policy = SACPolicy(observation_space, action_space)

        state = env.reset()

        action = policy.get_action(state)
        action_from_actor, _ = policy.actor(th.tensor(state, dtype=th.float32))

        assert action.shape == action_space.shape

    def test_get_action_deterministic(self, env):
        """Tests if get_action returns correct action in deterministic mode."""
        observation_space = env.observation_space
        action_space = env.action_space

        policy = SACPolicy(observation_space, action_space)

        state = env.reset()

        action = th.as_tensor(policy.get_action(state, deterministic=True), dtype=th.float32)
        action_from_actor, _ = policy.actor(th.as_tensor(state, dtype=th.float32), deterministic=True)

        assert action.shape == action_space.shape
        assert th.allclose(action, action_from_actor)
