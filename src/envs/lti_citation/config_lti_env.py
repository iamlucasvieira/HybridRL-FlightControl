"""Module that configures the LTI environment."""
import gym
from pydantic import BaseModel, validator, Extra
from typing import Optional, List, Literal

from envs.lti_citation.tasks import AVAILABLE_TASKS
from envs.lti_citation.rewards import AVAILABLE_REWARDS
from envs.lti_citation.observations import AVAILABLE_OBSERVATIONS
from envs.lti_citation.aircraft_environment import AircraftEnv


class ConfigLTISweep(BaseModel):
    """Class that makes it possible to built multiple sweep configurations."""
    reward_type: Optional[List[str]] = []
    observation_type: Optional[List[str]] = []
    task_type: Optional[List[str]] = []

    class Config:
        extra = Extra.forbid


class ConfigLTIKwargs(BaseModel):
    """Base configuration which is passed to the gym environment."""
    filename: Optional[str | List[str]] = "citation.yaml"
    configuration: Optional[str | List[str]] = "sp"
    dt: Optional[float | List[float]] = 0.1  # Time step
    episode_steps: Optional[int | List[int]] = 100  # Number of steps
    task_type: Optional[str | List[str]] = "sin_q"
    reward_scale: Optional[float | List[float]] = 1.0  # Reward scale
    reward_type: Optional[str | List[str]] = "sq_error"
    observation_type: Optional[str | List[str]] = "states + ref + error"

    @validator('configuration')
    def check_config(cls, configuration):
        CONFIGURATIONS = ['symmetric', 'sp', 'asymmetric']  # Available aircraft configurations
        if configuration not in CONFIGURATIONS and not isinstance(configuration, list):
            raise ValueError(f"Configuration must be in {CONFIGURATIONS}")
        return configuration

    @validator('task_type')
    def check_task_type(cls, task_type):
        if task_type not in AVAILABLE_TASKS and not isinstance(task_type, list):
            raise ValueError(f"Task must be in {AVAILABLE_TASKS}")
        return task_type

    @validator('reward_type')
    def check_reward_type(cls, reward_type):
        if reward_type not in AVAILABLE_REWARDS and not isinstance(reward_type, list):
            raise ValueError(f"Reward must be in {AVAILABLE_REWARDS}")
        return reward_type

    @validator('observation_type')
    def check_observation_type(cls, observation_type):
        if observation_type not in AVAILABLE_OBSERVATIONS and not isinstance(observation_type, list):
            raise ValueError(f"Observation must be in {AVAILABLE_OBSERVATIONS}")
        return observation_type

    class Config:
        extra = Extra.forbid


class ConfigLTIEnv(BaseModel):
    """Symmetric derivatives."""
    name: Literal['LTI'] = "LTI"
    kwargs: Optional[ConfigLTIKwargs] = ConfigLTIKwargs()
    sweep: Optional[ConfigLTIKwargs] = ConfigLTIKwargs()
    object: gym.Env = AircraftEnv

    class Config:
        extra = Extra.forbid
