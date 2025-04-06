"""Module that configures the LTI environment."""

from typing import List, Literal, Optional

import gymnasium as gym
from pydantic import BaseModel, ConfigDict, validator

from envs.citation.citation_env import CitationEnv
from envs.citation.models.model_loader import AVAILABLE_MODELS
from envs.observations import AVAILABLE_OBSERVATIONS
from envs.rewards import AVAILABLE_REWARDS
from tasks.all_tasks import AVAILABLE_TASKS


class ConfigCitationKwargs(BaseModel):
    """Base configuration which is passed to the gym environment."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    model: Optional[str | List[str]] = "default"
    dt: Optional[float | List[float]] = 0.01  # Time step
    episode_steps: Optional[int | List[int]] = 2_000  # Number of steps
    eval_steps: Optional[int | List[int]] = 2_000  # Number of steps
    reward_scale: Optional[float | List[float]] = 1.0  # Reward scale
    task_train: Optional[str | List[str]] = "att_train"
    task_eval: Optional[str | List[str]] = None
    reward_type: Optional[str | List[str]] = "sq_error"
    observation_type: Optional[str | List[str]] = "states + ref + error"
    input_names: Optional[List[str]] = None
    filter_action: Optional[bool] = False
    action_scale: Optional[List[float] | float] = 1

    @validator("model")
    def check_config(cls, model):
        if model not in AVAILABLE_MODELS and not isinstance(model, list):
            raise ValueError(f"Configuration must be in {model}")
        return model

    @validator("task_train")
    def check_reference_type(cls, task_train):
        if task_train not in AVAILABLE_TASKS and not isinstance(task_train, list):
            raise ValueError(f"Task must be in {AVAILABLE_TASKS}")
        return task_train

    @validator("task_eval")
    def check_eval_reference_type(cls, task_eval):
        if task_eval not in AVAILABLE_TASKS and not isinstance(task_eval, list):
            raise ValueError(f"Task must be in {AVAILABLE_TASKS}")
        return task_eval

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


class ConfigCitationEnv(BaseModel):
    """Symmetric derivatives."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: Literal["citation"] = "citation"
    kwargs: Optional[ConfigCitationKwargs] = ConfigCitationKwargs()
    sweep: Optional[ConfigCitationKwargs] = ConfigCitationKwargs()
    object: gym.Env = CitationEnv
