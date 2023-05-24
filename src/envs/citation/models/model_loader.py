"""Loads the model from the model directory"""
import importlib
from dataclasses import dataclass, fields
from typing import List, Optional


@dataclass
class Model:
    """Base class for models."""

    name: str
    inputs: List[str]
    states: List[str]
    n_inputs: Optional[int] = None
    n_states: Optional[int] = None

    def __post_init__(self):
        object.__setattr__(self, "n_inputs", len(self.inputs))
        object.__setattr__(self, "n_states", len(self.states))


models_dict = {
    "default": Model(
        name="default",
        inputs=[
            "de",
            "da",
            "dr",
            "de_t",
            "da_t",
            "dr_t",
            "df",
            "gear",
            "throttle_1",
            "throttle_2",
        ],
        states=[
            "p",
            "q",
            "r",
            "V",
            "alpha",
            "beta",
            "phi",
            "theta",
            "psi",
            "he",
            "xe",
            "ye",
        ],
    ),
}

AVAILABLE_MODELS = models_dict.keys()


def load_model(model_name: str):
    """Loads a model from the models directory."""
    if model_name in AVAILABLE_MODELS:
        model = importlib.import_module(f"envs.citation.models.{model_name}.citation")
        for field in fields(models_dict[model_name]):
            setattr(model, field.name, getattr(models_dict[model_name], field.name))
        return model
    else:
        raise ValueError(f"Model {model_name} not available.")
