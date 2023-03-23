import pytest

from envs import CitationEnv, LTIEnv
from helpers.torch_helpers import BaseNetwork


@pytest.mark.parametrize("env", [LTIEnv, CitationEnv])
class TestBaseNetwork:
    """Test BaseNetwork class for functionalities that are common to Actor and Critic."""

    def test_abstract(self, env):
        """Test that BaseNetwork is an abstract class and child is correctly initialized."""
        env = env()
        observation_space = env.observation_space
        action_space = env.action_space
        with pytest.raises(NotImplementedError):
            BaseNetwork(observation_space, action_space)
