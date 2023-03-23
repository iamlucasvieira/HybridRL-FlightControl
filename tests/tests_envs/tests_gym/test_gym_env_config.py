import pytest
from pydantic import ValidationError

from envs.config.gym_env import ConfigGymEnv, ConfigGymKwargs


class TestConfigGymEnv:
    """Test the gym environment configuration."""

    def test_init(self):
        """Test the initialization of the configuration."""

        kwargs = ConfigGymKwargs(id="CartPole-v1")
        config = ConfigGymEnv(kwargs=kwargs)
        assert config.kwargs.id == "CartPole-v1"

    def test_invalid_extra(self):
        """Test the initialization of the configuration with an invalid extra."""

        with pytest.raises(ValidationError):
            ConfigGymEnv(invalid=True)

    def test_missing_kwargs(self):
        """Test the initialization of the configuration with an invalid extra."""

        with pytest.raises(ValidationError):
            ConfigGymEnv()


class TestConfigGymKwargs:
    """Test the gym environment configuration."""

    def test_init(self):
        """Test the initialization of the configuration."""

        kwargs = ConfigGymKwargs(id="CartPole-v1")
        assert kwargs.id == "CartPole-v1"

    def test_init_invalid_id(self):
        """Test the initialization of the configuration with an invalid id."""

        with pytest.raises(ValidationError):
            ConfigGymKwargs(id="Invalid")

    def test_invalid_extra(self):
        """Test the initialization of the configuration with an invalid extra."""

        with pytest.raises(ValidationError):
            ConfigGymKwargs(invalid=True)

    def test_sweep_not_none(self):
        """Test the initialization with a valid sweep."""

        kwargs = ConfigGymKwargs(id="CartPole-v1")
        config = ConfigGymEnv(kwargs=kwargs)
        assert config.sweep.id == "CartPole-v1"
