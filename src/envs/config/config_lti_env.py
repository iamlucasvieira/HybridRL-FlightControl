"""Module that configures the LTI environment."""
import gym
from pydantic import BaseModel, validator, Extra
from typing import Optional, List, Literal

from envs.reference_signals import AVAILABLE_REFERENCES
from envs.rewards import AVAILABLE_REWARDS
from envs.observations import AVAILABLE_OBSERVATIONS
from envs.lti_citation.lti_env import LTIEnv


class ConfigLTIKwargs(BaseModel):
    """Base configuration which is passed to the gym environment."""
    filename: Optional[str | List[str]] = "citation.yaml"
    configuration: Optional[str | List[str]] = "sp"
    dt: Optional[float | List[float]] = 0.1  # Time step
    episode_steps: Optional[int | List[int]] = 100  # Number of steps
    reward_scale: Optional[float | List[float]] = 1.0  # Reward scale
    reference_type: Optional[str | List[str]] = "sin"
    reward_type: Optional[str | List[str]] = "sq_error"
    observation_type: Optional[str | List[str]] = "states + ref + error"
    tracked_state: Optional[str | List[str]] = "q"

    @validator('configuration')
    def check_config(cls, configuration):
        CONFIGURATIONS = ['symmetric', 'sp', 'asymmetric']  # Available aircraft configurations
        if configuration not in CONFIGURATIONS and not isinstance(configuration, list):
            raise ValueError(f"Configuration must be in {CONFIGURATIONS}")
        return configuration

    @validator('reference_type')
    def check_reference_type(cls, reference_type):
        if reference_type not in AVAILABLE_REFERENCES and not isinstance(reference_type, list):
            raise ValueError(f"Task must be in {AVAILABLE_REFERENCES}")
        return reference_type

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
    object: gym.Env = LTIEnv

    class Config:
        extra = Extra.forbid