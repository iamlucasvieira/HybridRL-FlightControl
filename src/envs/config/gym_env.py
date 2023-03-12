"""Gym environment configuration."""
import gym
from pydantic import BaseModel, Extra, validator
from typing import Optional, List, Literal, Callable


class ConfigGymKwargs(BaseModel):
    """Base configuration which is passed to the gym environment."""
    id: str

    class Config:
        extra = Extra.forbid

    @validator('id')
    def check_id(cls, id):
        if id not in gym.envs.registry.env_specs:
            raise ValueError(f"Environment {id} is not registered in gym.")
        return id


class ConfigGymEnv(BaseModel):
    """Symmetric derivativeis."""
    name: Literal['gym'] = 'gym'
    kwargs: ConfigGymKwargs
    sweep: Optional[ConfigGymKwargs] = None
    object: Optional[Callable] = gym.make

    class Config:
        extra = Extra.forbid

    @validator('sweep', always=True)
    def check_sweep(cls, sweep, values):
        if sweep is None and 'kwargs' in values:
            return values['kwargs']
        return sweep
