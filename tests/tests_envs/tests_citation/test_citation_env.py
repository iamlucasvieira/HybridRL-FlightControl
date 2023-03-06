import pytest
import numpy as np

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

        first_response, _, _, _ = env.step(np.array([0] * env.action_space.shape[0]))
        env.reset()
        second_response, _, _, _ = env.step(np.array([0] * env.action_space.shape[0]))

        assert (first_response == second_response).all(), f"env.model should be reset after an environment reset"

    def test_unsupported_model(self, env_kwargs):
        """Tests if unsupported model raises an error."""
        env_kwargs["model"] = "unsupported"
        with pytest.raises(ValueError):
            CitationEnv(**env_kwargs)
