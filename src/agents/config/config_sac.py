"""Module with SAC configuration."""
from pydantic import BaseModel, Extra
from typing import Optional, Literal
from stable_baselines3 import SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from helpers.config_auto import get_auto


class ConfigSACArgs(BaseModel):
    """Arguments for IDHP object."""
    policy: Optional[str] = "MlpPolicy"
    env: Optional[str] = get_auto("env")

    class Config:
        extra = Extra.forbid


class ConfigSACKwargs(BaseModel):
    """Keyword arguments for IDHP object."""
    pass

    class Config:
        extra = Extra.forbid


class ConfigSACLearn(BaseModel):
    """Allows defining parameters that can be passed to learn method."""
    total_timesteps: Optional[int] = 1_000
    log_interval: Optional[int] = 1
    progress_bar: Optional[bool] = False
    callback: Optional[list] = ["tensorboard"]

    class Config:
        extra = Extra.forbid


# Configuration of Agents
class ConfigSAC(BaseModel):
    """Configuration of SAC."""
    name: Literal['SAC'] = "SAC"
    args: Optional[ConfigSACArgs] = ConfigSACArgs()
    kwargs: Optional[ConfigSACKwargs] = ConfigSACKwargs()
    sweep: Optional[ConfigSACKwargs] = ConfigSACKwargs()
    learn: Optional[ConfigSACLearn] = ConfigSACLearn()
    object: BaseAlgorithm = SAC

    class Config:
        extra = Extra.forbid
