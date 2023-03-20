import numpy as np
import pytest

from envs.lti_citation.lti_env import LTIEnv


@pytest.fixture
def env_kwargs():
    default_kwargs = {
        "filename": "citation.yaml",
        "configuration": "sp",
        "dt": 0.1,
        "episode_steps": 100,
        "tracked_state": "q",
        "reference_type": "sin",
        "reward_type": "sq_error",
        "observation_type": "states + ref + error",
        "reward_scale": 1.0,
    }
    return default_kwargs


class TestLTIEnv:
    """Tests LTIEnv."""

    def test_aircraft_reset(self, env_kwargs):
        """Tests if aircraft is reset after a reset."""
        env = LTIEnv(**env_kwargs)

        env.step(np.array([0]))
        assert (
            len(env.aircraft.states) == 2
        ), f"env.aircraft.states should be updated after an environmnet step"
        env.reset()
        assert (
            len(env.aircraft.states) == 1
        ), f"env.aircraft.states should be reset after an environmnet reset"

    def test_unsupported_configuration(self, env_kwargs):
        """Tests if unsupported configuration raises an error."""
        env_kwargs["configuration"] = "unsupported"
        with pytest.raises(ValueError):
            LTIEnv(**env_kwargs)
