"""Module with SAC configuration."""
from pydantic import BaseModel, Extra
from typing import Optional, Literal, List
from stable_baselines3 import SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from helpers.config_auto import get_auto


class ConfigSB3SACArgs(BaseModel):
    """Arguments for IDHP object."""
    policy: Optional[str] = "MlpPolicy"
    env: Optional[str] = get_auto("env")

    class Config:
        extra = Extra.forbid


class ConfigSB3SACKwargs(BaseModel):
    """Keyword arguments for IDHP object."""
    verbose: Optional[int | List[int]] = get_auto("verbose")

    class Config:
        extra = Extra.forbid


class ConfigSB3SACLearn(BaseModel):
    """Allows defining parameters that can be passed to learn method."""
    total_timesteps: Optional[int] = 1_000
    log_interval: Optional[int] = 1
    progress_bar: Optional[bool] = False
    callback: Optional[list] = ["tensorboard"]

    class Config:
        extra = Extra.forbid


# Configuration of Agents
class ConfigSB3SAC(BaseModel):
    """Configuration of SAC."""
    name: Literal['SB3SAC'] = "SB3SAC"
    args: Optional[ConfigSB3SACArgs] = ConfigSB3SACArgs()
    kwargs: Optional[ConfigSB3SACKwargs] = ConfigSB3SACKwargs()
    sweep: Optional[ConfigSB3SACKwargs] = ConfigSB3SACKwargs()
    learn: Optional[ConfigSB3SACLearn] = ConfigSB3SACLearn()
    object: BaseAlgorithm = SAC

    class Config:
        extra = Extra.forbid
