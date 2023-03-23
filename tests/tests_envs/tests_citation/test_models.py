import pytest

from envs.citation.models.model_loader import AVAILABLE_MODELS, load_model


class TestModelLoader:
    @pytest.mark.parametrize("model_name", AVAILABLE_MODELS)
    def test_load_model(self, model_name):
        """Test that the model is loaded correctly."""
        model = load_model(model_name)
        assert model.name == model_name
        assert len(model.inputs) == model.n_inputs
        assert len(model.states) == model.n_states

    def test_unavailable_model(self):
        """Test that the model is loaded correctly."""
        with pytest.raises(ValueError):
            load_model("unavailable_model")
