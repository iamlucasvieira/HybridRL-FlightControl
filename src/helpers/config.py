"""Module that define configuration of algorithms."""
from pydantic import BaseModel, validator
import numpy as np
from typing import Optional, List

from models.tasks import AVAILABLE_TASKS
from models.rewards import AVAILABLE_REWARDS
from models.observations import AVAILABLE_OBSERVATIONS


class ConfigLinearAircraft(BaseModel):
    """Symmetric derivatives."""
    policy_type: Optional[str] = "MlpPolicy"
    env_name: Optional[str] = "citation"
    filename: Optional[str] = "citation.yaml"
    configuration: Optional[str] = "symmetric"
    algorithm: Optional[str] = ""
    seed: Optional[int] = None  # Random seed
    dt: Optional[float] = 0.1  # Time step
    episode_steps: Optional[int] = 100  # Number of steps
    learning_steps: Optional[int] = 1_000  # Number of total learning steps
    task_type: Optional[str] = "sin_q"
    evaluate: Optional[int] = 1  # Number of times to run the environment after learning
    reward_scale: Optional[float] = 1.0  # Reward scale
    log_interval: Optional[int] = 1  # Log interval
    reward_type: Optional[str] = "sq_error"
    observation_type: Optional[str] = "states + ref + error"

    @validator('configuration')
    def check_config(cls, configuration):
        CONFIGURATIONS = ['symmetric', 'sp', 'asymmetric']  # Available aircraft configurations
        if configuration not in CONFIGURATIONS:
            raise ValueError(f"Configuration must be in {CONFIGURATIONS}")
        return configuration

    @validator('task_type')
    def check_task_type(cls, task):
        if task_type not in AVAILABLE_TASKS:
            raise ValueError(f"Task must be in {AVAILABLE_TASKS}")
        return task_type

    @validator('reward_type')
    def check_reward_type(cls, reward_type):
        if reward_type not in AVAILABLE_REWARDS:
            raise ValueError(f"Reward must be in {AVAILABLE_REWARDS}")
        return reward_type

    @validator('observation_type')
    def check_observation_type(cls, observation_type):
        if observation_type not in AVAILABLE_OBSERVATIONS:
            raise ValueError(f"Observation must be in {AVAILABLE_OBSERVATIONS}")
        return observation_type

    @validator('seed')
    def check_seed(cls, seed):
        if seed is None:
            seed = np.random.randint(1_000)
        return seed


class ConfigExperiment(BaseModel):
    """Class that defines the configuration of the sweep."""
    project_name: Optional[str]
    offline: Optional[bool] = False
    n_learning: Optional[int] = 1
    config: Optional[ConfigLinearAircraft] = ConfigLinearAircraft()
    algorithm: Optional[List[str]] = []
    reward_type: Optional[List[str]] = []
    observation_type: Optional[List[str]] = []
