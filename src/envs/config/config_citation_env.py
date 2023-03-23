"""Module that configures the LTI environment."""
from typing import List, Literal, Optional

import gymnasium as gym
from pydantic import BaseModel, Extra, validator

from envs.citation.citation_env import CitationEnv
from envs.citation.models.model_loader import AVAILABLE_MODELS
from envs.observations import AVAILABLE_OBSERVATIONS
from envs.reference_signals import AVAILABLE_REFERENCES
from envs.rewards import AVAILABLE_REWARDS


class ConfigCitationKwargs(BaseModel):
    """Base configuration which is passed to the gym environment."""

    model: Optional[str | List[str]] = "default"
    dt: Optional[float | List[float]] = 0.1  # Time step
    episode_steps: Optional[int | List[int]] = 100  # Number of steps
    reward_scale: Optional[float | List[float]] = 1.0  # Reward scale
    reference_type: Optional[str | List[str]] = "sin"
    reward_type: Optional[str | List[str]] = "sq_error"
    observation_type: Optional[str | List[str]] = "states + ref + error"
    tracked_state: Optional[str | List[str]] = "q"

    @validator("model")
    def check_config(cls, model):
        if model not in AVAILABLE_MODELS and not isinstance(model, list):
            raise ValueError(f"Configuration must be in {model}")
        return model

    @validator("reference_type")
    def check_reference_type(cls, reference_type):
        if reference_type not in AVAILABLE_REFERENCES and not isinstance(
            reference_type, list
        ):
            raise ValueError(f"Task must be in {AVAILABLE_REFERENCES}")
        return reference_type

    @validator("reward_type")
    def check_reward_type(cls, reward_type):
        if reward_type not in AVAILABLE_REWARDS and not isinstance(reward_type, list):
            raise ValueError(f"Reward must be in {AVAILABLE_REWARDS}")
        return reward_type

    @validator("observation_type")
    def check_observation_type(cls, observation_type):
        if observation_type not in AVAILABLE_OBSERVATIONS and not isinstance(
            observation_type, list
        ):
            raise ValueError(f"Observation must be in {AVAILABLE_OBSERVATIONS}")
        return observation_type

    class Config:
        extra = Extra.forbid


class ConfigCitationEnv(BaseModel):
    """Symmetric derivatives."""

    name: Literal["citation"] = "citation"
    kwargs: Optional[ConfigCitationKwargs] = ConfigCitationKwargs()
    sweep: Optional[ConfigCitationKwargs] = ConfigCitationKwargs()
    object: gym.Env = CitationEnv

    class Config:
        extra = Extra.forbid
