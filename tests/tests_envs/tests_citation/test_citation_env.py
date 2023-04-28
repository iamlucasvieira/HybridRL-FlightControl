import numpy as np
import pytest

from envs.citation.citation_env import CitationEnv


@pytest.fixture
def env_kwargs():
    default_kwargs = {
        "model": "default",
        "dt": 0.1,
        "episode_steps": 100,
        "tracked_state": "q",
        "reference_type": "sin",
        "reward_type": "sq_error",
        "observation_type": "states + ref + error",
        "reward_scale": 1.0,
    }
    return default_kwargs


class TestCitationEnv:
    """Tests CitationEnv."""

    def test_aircraft_reset(self, env_kwargs):
        """Tests if aircraft is reset after a reset."""
        env = CitationEnv(**env_kwargs)

        first_response, _, _, _, _ = env.step(np.array([0] * env.action_space.shape[0]))
        env.reset()
        second_response, _, _, _, _ = env.step(
            np.array([0] * env.action_space.shape[0])
        )

        assert (
            first_response == second_response
        ).all(), f"env.model should be reset after an environment reset"

    def test_unsupported_model(self, env_kwargs):
        """Tests if unsupported model raises an error."""
        env_kwargs["model"] = "unsupported"
        with pytest.raises(ValueError):
            CitationEnv(**env_kwargs)

    def test_unsupported_tracking_state(self, env_kwargs):
        """Tests if unsupported tracking state raises an error."""
        env_kwargs["tracked_state"] = "unsupported"
        with pytest.raises(ValueError):
            CitationEnv(**env_kwargs)

    def test_init_with_input_names(self, env_kwargs):
        """Tests if the environment can be initialized with input_names."""
        input_names = ["de", "da"]
        env_kwargs["input_names"] = input_names
        env = CitationEnv(**env_kwargs)
        for input_name in input_names:
            assert input_name in env.input_names
        assert len(env.input_idx) == len(input_names)
        assert env.action_space.shape == (2,)

    def test_init_unsupported_input_names(self, env_kwargs):
        """Tests if unsupported input_names raises an error."""
        env_kwargs["input_names"] = ["unsupported"]
        with pytest.raises(ValueError):
            CitationEnv(**env_kwargs)

    def test_init_with_observation_names(self, env_kwargs):
        """Tests if the environment can be initialized with observation_names."""
        observation_names = ["p", "q", "r"]
        env_kwargs["observation_names"] = observation_names
        env = CitationEnv(**env_kwargs)
        for observation in observation_names:
            assert observation in env.observation_names
        assert len(env.observation_idx) == len(observation_names)

    def test_init_unsupported_observation_names(self, env_kwargs):
        """Tests if unsupported observation_names raises an error."""
        env_kwargs["observation_names"] = ["unsupported"]
        with pytest.raises(ValueError):
            CitationEnv(**env_kwargs)

    def test_init_with_default_input_observation(self, env_kwargs):
        """Tests if the environment can be initialized with default input and observation names."""
        env = CitationEnv(**env_kwargs)
        assert len(env.input_idx) == 3
        assert len(env.observation_idx) == 7
        assert env.action_space.shape[0] == 3

    def test_list_contains_all(self):
        """Test the method list_contains_all."""
        assert CitationEnv.list_contains_all([1, 2, 3], [1, 2])
        assert CitationEnv.list_contains_all([1, 2, 3], [1, 2, 3])
        assert CitationEnv.list_contains_all(["a", "b", "c"], ["a", "b", "c"])
        assert CitationEnv.list_contains_all(["a", "b", "c"], ["a", "b"])

    def test_list_contains_all_error(self):
        """Test the method list_contains_all when list does not contain an item."""
        with pytest.raises(ValueError):
            CitationEnv.list_contains_all([1, 2, 3], [1, 2, 4])
        with pytest.raises(ValueError):
            CitationEnv.list_contains_all([], [1, 4, 3])
        with pytest.raises(ValueError):
            CitationEnv.list_contains_all(["a", "b", "c"], ["a", "b", "d"])

    def test_action_to_input(self, env_kwargs):
        """Test the method action_to_input."""
        env = CitationEnv(**env_kwargs)

        action = np.array([0.1, 0.2, 0.3])
        input_values = env.action_to_input(action)
        model_n_inputs = env.model.n_inputs

        assert input_values.shape == (model_n_inputs,)
        for value, idx in zip(action, env.input_idx):
            assert value == input_values[idx]
