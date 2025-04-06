"""Gym environment configuration."""

from typing import Callable, Literal, Optional

import gymnasium as gym
from pydantic import BaseModel, ConfigDict, field_validator


class ConfigGymKwargs(BaseModel):
    """Base configuration which is passed to the gym environment."""

    model_config = ConfigDict(extra="forbid")

    id: str
    render_mode: Optional[str] = "human"

    @field_validator("id")
    def check_id(cls, id):
        if id not in gym.envs.registry.keys():
            raise ValueError(f"Environment {id} is not registered in gym.")
        return id


class ConfigGymEnv(BaseModel):
    """Symmetric derivativeis."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["gym"] = "gym"
    kwargs: ConfigGymKwargs
    sweep: Optional[ConfigGymKwargs] = None
    object: Optional[Callable] = gym.make

    @field_validator("sweep")
    def check_sweep(cls, sweep, values):
        if sweep is None and "kwargs" in values:
            return values["kwargs"]
        return sweep
