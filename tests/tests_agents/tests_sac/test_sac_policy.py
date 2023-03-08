import pytest

from agents.sac.policy import BaseNetwork, CriticNetwork, ActorNetwork
from envs.citation.citation_env import CitationEnv
from torch import nn


@pytest.fixture
def env():
    """Create a CitationEnv instance."""
    return CitationEnv()


@pytest.mark.parametrize("network", [CriticNetwork])
class TestBaseNetwork:
    """Test BaseNetwork class for functionalities that are common to Actor and Critic."""

    def test_abstract(self, env, network):
        """Test that BaseNetwork is an abstract class and child is correctly initialized."""
        observation_space = env.observation_space
        action_space = env.action_space
        with pytest.raises(NotImplementedError):
            BaseNetwork(observation_space, action_space)

        net = network(observation_space, action_space)
        assert net is not None

    def test_shapes(self, env, network):
        """Test if network creates correct input and observation shapes."""
        observation_space = env.observation_space
        action_space = env.action_space

        net = network(observation_space, action_space)

        assert net.observation_dim == observation_space.shape[0]
        assert net.action_dim == action_space.shape[0]


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
