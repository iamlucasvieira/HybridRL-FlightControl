import numpy as np
import pytest

from envs.citation.citation_env import CitationEnv
from envs.lti_citation.lti_env import LTIEnv
from envs.observations import AVAILABLE_OBSERVATIONS
from envs.observations import get_observation
from envs.reference_signals import AVAILABLE_REFERENCES
from envs.reference_signals import get_reference_signal
from envs.rewards import AVAILABLE_REWARDS
from envs.rewards import get_reward


env_storage_parameters = [
    "actions",
    "reference",
    "track",
    "states",
    "error",
    "sq_error",
]


@pytest.fixture
def env_kwargs():
    default_kwargs = {
        "dt": 0.1,
        "episode_steps": 100,
        "reward_scale": 1.0,
        "tracked_state": "q",
        "reference_type": "sin",
        "reward_type": "sq_error",
        "observation_type": "states + ref + error",
    }
    return default_kwargs


@pytest.mark.parametrize("env_constructor", [LTIEnv, CitationEnv])
class TestChildEnvs:
    """Tests the initialization of child envs of BaseEnv.

    This tests common functionalities of LTIEnv and CitationEnv.
    """

    def test_init(self, env_constructor, env_kwargs):
        """Tests if environment is created."""
        env = env_constructor(**env_kwargs)
        assert env is not None

    @pytest.mark.parametrize("parameter", env_storage_parameters)
    def test_ini_storage(self, env_constructor, parameter, env_kwargs):
        """Tests if parameters used to store values start with initial value."""
        env = env_constructor(**env_kwargs)
        assert (
            len(getattr(env, parameter)) == 1
        ), f"env.{parameter} should not be an empty list."

    @pytest.mark.parametrize("parameter", env_storage_parameters)
    def test_step_storage(self, env_constructor, parameter, env_kwargs):
        """Tests if parameters used to store value are updated after a step."""
        env = env_constructor(**env_kwargs)
        env.step(np.array([0] * env.action_space.shape[0]))
        assert (
            len(getattr(env, parameter)) == 2
        ), f"env.{parameter} should be updated after an environmnet step"

    @pytest.mark.parametrize("parameter", env_storage_parameters)
    def test_reset_storage(self, env_constructor, parameter, env_kwargs):
        """Tests if parameters used to store value are reset after a reset."""
        env = env_constructor(**env_kwargs)
        env.step(np.array([0] * env.action_space.shape[0]))
        env.reset()
        assert (
            len(getattr(env, parameter)) == 1
        ), f"env.{parameter} should be reset after an environmnet reset"

    @pytest.mark.parametrize("observation_type", AVAILABLE_OBSERVATIONS)
    def test_observation_space(self, env_constructor, observation_type, env_kwargs):
        """Tests if observation space is correct."""
        env_kwargs["observation_type"] = observation_type
        env = env_constructor(**env_kwargs)

        obs_function_shape = env.get_obs(env).shape
        observation_space_shape = env.observation_space.shape
        assert (
            obs_function_shape == observation_space_shape
        ), f"Observation space shape {observation_space_shape} does not match observation function shape {obs_function_shape}."
        assert env.get_obs == get_observation(
            observation_type
        ), f"Observation function is not set correctly."

    def test_invalid_observation_type(self, env_constructor, env_kwargs):
        """Tests if error is raised when invalid observation type is given."""
        env_kwargs["observation_type"] = "invalid"
        with pytest.raises(ValueError):
            env_constructor(**env_kwargs)

    @pytest.mark.parametrize("reward_type", AVAILABLE_REWARDS)
    def test_reward_function(self, env_constructor, reward_type, env_kwargs):
        """Tests if reward function is correct."""
        env_kwargs["reward_type"] = reward_type
        env = env_constructor(**env_kwargs)
        reward = env.get_reward(env)
        assert reward == 0, "Reward should be 0 at the beginning of the episode."
        assert env.get_reward == get_reward(
            reward_type
        ), f"Reward function is not set correctly."

    def test_invalid_reward_type(self, env_constructor, env_kwargs):
        """Tests if error is raised when invalid reward type is given."""
        env_kwargs["reward_type"] = "invalid"
        with pytest.raises(ValueError):
            env_constructor(**env_kwargs)

    @pytest.mark.parametrize("reference_type", AVAILABLE_REFERENCES)
    def test_reference_function(self, env_constructor, reference_type, env_kwargs):
        """Tests if reference function is correct."""
        env_kwargs["reference_type"] = reference_type
        env = env_constructor(**env_kwargs)
        reference = env.get_reference(env)
        assert isinstance(reference, np.ndarray), "Reference should be a numpy array."
        assert env.get_reference == get_reference_signal(
            reference_type
        ), f"Reference function is not set correctly."

    def test_invalid_reference_type(self, env_constructor, env_kwargs):
        """Tests if error is raised when invalid reference type is given."""
        env_kwargs["reference_type"] = "invalid"
        with pytest.raises(ValueError):
            env_constructor(**env_kwargs)

    @pytest.mark.parametrize("reference_type", AVAILABLE_REFERENCES)
    def test_set_reference(self, env_constructor, reference_type, env_kwargs):
        """Tests if using set_reference_signal changes the reference function"""
        env = env_constructor(**env_kwargs)
        env.set_reference_signal(reference_type)
        assert env.get_reference == get_reference_signal(
            reference_type
        ), f"Reference function is not set correctly."

    @pytest.mark.parametrize("reward_type", AVAILABLE_REWARDS)
    def test_set_reward(self, env_constructor, reward_type, env_kwargs):
        """Tests if using set_reward_function changes the reward function"""
        env = env_constructor(**env_kwargs)
        env.set_reward_function(reward_type)
        assert env.get_reward == get_reward(
            reward_type
        ), f"Reward function is not set correctly."

    @pytest.mark.parametrize("observation_type", AVAILABLE_OBSERVATIONS)
    def test_set_observation(self, env_constructor, observation_type, env_kwargs):
        """Tests if using set_observation_function changes the observation function"""
        env = env_constructor(**env_kwargs)
        env.set_observation_function(observation_type)
        obs_function_shape = env.get_obs(env).shape
        observation_space_shape = env.observation_space.shape
        assert env.get_obs == get_observation(
            observation_type
        ), f"Observation function is not set correctly."
        assert (
            obs_function_shape == observation_space_shape
        ), f"Observation space shape {observation_space_shape} does not match observation function shape {obs_function_shape}."

    def test_shapes(self, env_constructor, env_kwargs):
        """Tests if the shapes of the state, action and observation spaces are correct."""
        env = env_constructor(**env_kwargs)
        for i in range(10):
            shape_tracked_state = (np.sum(env.tracked_state_mask),)
            assert np.isclose(env.current_time, env.dt * i)
            assert env.actions[i].shape == env.action_space.shape
            assert env.error[i].shape == shape_tracked_state
            assert env.sq_error[i].shape == shape_tracked_state
            assert env.states[i].shape == (env.n_states,)
            assert env.reference[i].shape == shape_tracked_state
            assert env.track[i].shape == shape_tracked_state
            env.step(env.action_space.sample())
