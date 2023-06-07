"""Module that contains the tests for the IDHP incremental model."""
import pytest

from agents.idhp.incremental_model import IncrementalCitation
from envs import BaseEnv, CitationEnv, LTIEnv


@pytest.mark.parametrize("env", [CitationEnv, LTIEnv])
class TestIncrementalCitation:
    """Tests for the IncrementalCitation class."""

    def test_init(self, env):
        """Tests the init method."""
        env = env()
        n_states = env.n_states
        n_inputs = env.n_inputs
        model = IncrementalCitation(env, states_mask=[1] * n_states)
        assert model.n_inputs == n_inputs
        assert model.n_states == n_states
        assert model.cov.shape == (n_inputs + n_states, n_inputs + n_states)
        assert model.theta.shape == (n_inputs + n_states, n_states)
        assert len(model.errors) == 1
        assert model.F is not None
        assert model.G is not None

    def test_increment_not_enough_data(self, env):
        """Tests the increment method."""
        env = env()
        n_states = env.n_states
        model = IncrementalCitation(env, states_mask=[1] * n_states)

        assert not model.ready
        with pytest.raises(ValueError):
            model.increment(env)

    def test_increment(self, env):
        """Tests the increment method."""
        env = env()
        n_states = env.n_states
        model = IncrementalCitation(env, states_mask=[1] * n_states)
        env.step(env.action_space.sample())
        model.increment(env)
        assert model.ready

    def test_update_not_ready(self, env):
        """Test the update method when not ready to update."""
        env = env()
        n_states = env.n_states
        model = IncrementalCitation(env, states_mask=[1] * n_states)
        env.step(env.action_space.sample())
        assert not model.ready
        model.update(env)
        assert model.ready

    def test_update(self, env):
        """Test the update method."""
        env = env()
        n_states = env.n_states
        model = IncrementalCitation(env, states_mask=[1] * n_states)
        for _ in range(2):
            env.step(env.action_space.sample())
            model.update(env)
        assert len(model.errors) == 2

    def test_predict(self, env):
        """Test the predict method."""
        env = env()
        n_states = env.n_states
        model = IncrementalCitation(env, states_mask=[1] * n_states)
        env.step(env.action_space.sample())
        model.update(env)

        prediction = model.predict(
            model.state_k, model.state_k_1, model.action_k, model.action_k_1
        )
        assert prediction.shape == (model.n_states, 1)

    def test_predict_increment(self, env):
        """Test the predict method also returning the increment."""
        env = env()
        n_states = env.n_states
        model = IncrementalCitation(env, states_mask=[1] * n_states)
        env.step(env.action_space.sample())
        model.update(env)

        dx_hat, X = model.predict(
            model.state_k,
            model.state_k_1,
            model.action_k,
            model.action_k_1,
            increment=True,
        )
        assert dx_hat.shape == (model.n_states, 1)
        assert X.shape == (model.n_states + model.n_inputs, 1)
