"""Module that configures the LTI environment."""

from typing import List, Literal, Optional

import gymnasium as gym
from pydantic import BaseModel, ConfigDict, field_validator

from envs.lti_citation.lti_env import LTIEnv
from envs.observations import AVAILABLE_OBSERVATIONS
from envs.rewards import AVAILABLE_REWARDS
from tasks.all_tasks import AVAILABLE_TASKS


class ConfigLTIKwargs(BaseModel):
    """Base configuration which is passed to the gym environment."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    filename: Optional[str | List[str]] = "citation.yaml"
    configuration: Optional[str | List[str]] = "sp"
    dt: Optional[float | List[float]] = 0.1  # Time step
    episode_steps: Optional[int | List[int]] = 100  # Number of steps
    eval_steps: Optional[int | List[int]] = 100  # Number of steps
    reward_scale: Optional[float | List[float]] = 1.0  # Reward scale
    task_train: Optional[str | List[str]] = "sin_q"
    task_eval: Optional[str | List[str]] = None
    reward_type: Optional[str | List[str]] = "sq_error"
    observation_type: Optional[str | List[str]] = "states + ref + error"

    @field_validator("configuration")
    def check_config(cls, configuration):
        CONFIGURATIONS = [
            "symmetric",
            "sp",
            "asymmetric",
        ]  # Available aircraft configurations
        if configuration not in CONFIGURATIONS and not isinstance(configuration, list):
            raise ValueError(f"Configuration must be in {CONFIGURATIONS}")
        return configuration

    @field_validator("task_train")
    def check_task_type(cls, task_train):
        if task_train not in AVAILABLE_TASKS and not isinstance(task_train, list):
            raise ValueError(f"Task must be in {AVAILABLE_TASKS}")
        return task_train

    @field_validator("task_eval")
    def check_eval_task_type(cls, task_eval):
        if task_eval not in AVAILABLE_TASKS and not isinstance(task_eval, list):
            raise ValueError(f"Task must be in {AVAILABLE_TASKS}")
        return task_eval

    @field_validator("reward_type")
    def check_reward_type(cls, reward_type):
        if reward_type not in AVAILABLE_REWARDS and not isinstance(reward_type, list):
            raise ValueError(f"Reward must be in {AVAILABLE_REWARDS}")
        return reward_type

    @field_validator("observation_type")
    def check_observation_type(cls, observation_type):
        if observation_type not in AVAILABLE_OBSERVATIONS and not isinstance(
            observation_type, list
        ):
            raise ValueError(f"Observation must be in {AVAILABLE_OBSERVATIONS}")
        return observation_type


class ConfigLTIEnv(BaseModel):
    """Symmetric derivatives."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: Literal["LTI"] = "LTI"
    kwargs: Optional[ConfigLTIKwargs] = ConfigLTIKwargs()
    sweep: Optional[ConfigLTIKwargs] = ConfigLTIKwargs()
    object: gym.Env = LTIEnv
