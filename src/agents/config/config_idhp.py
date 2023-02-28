"""Module with IHDP configuration."""
from pydantic import BaseModel
from typing import Optional, Literal
from stable_baselines3.common.base_class import BaseAlgorithm
from agents.idhp import IDHP
from helpers.config_auto import get_auto


class ConfigIDHPArgs(BaseModel):
    """Arguments for IDHP object."""
    policy: Optional[str] = "default"
    env: Optional[str] = get_auto("env")


class ConfigIDHPKwargs(BaseModel):
    """Keyword arguments for IDHP object."""
    verbose: Optional[int] = get_auto("verbose")
    tensorboard_log: Optional[str] = get_auto("tensorboard_log")
    seed: Optional[int] = get_auto("seed")


class ConfigIDHPSweep(BaseModel):
    """Allows defining parameters that can be swept over."""
    pass


class ConfigIDHPLearn(BaseModel):
    """Allows defining parameters that can be passed to learn method."""
    total_timesteps: Optional[int] = 1_000
    log_interval: Optional[int] = 1
    progress_bar: Optional[bool] = False
    callback: Optional[list] = ["online", "tensorboard"]


class ConfigIDHP(BaseModel):
    name: Optional[Literal['IDHP']] = "IDHP"
    args: Optional[ConfigIDHPArgs] = ConfigIDHPArgs()
    kwargs: Optional[ConfigIDHPKwargs] = ConfigIDHPKwargs()
    sweep: Optional[ConfigIDHPSweep] = ConfigIDHPSweep()
    learn: Optional[ConfigIDHPLearn] = ConfigIDHPLearn()
    object: BaseAlgorithm = IDHP
