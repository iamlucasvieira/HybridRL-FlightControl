"""Module with IHDP configuration."""
from pydantic import BaseModel, Extra
from typing import Optional, Literal, List
from stable_baselines3.common.base_class import BaseAlgorithm
from agents.idhp import IDHP
from helpers.config_auto import get_auto


class ConfigIDHPArgs(BaseModel):
    """Arguments for IDHP object."""
    policy: Optional[str] = "default"
    env: Optional[str] = get_auto("env")

    class Config:
        extra = Extra.forbid


class ConfigIDHPKwargs(BaseModel):
    """Keyword arguments for IDHP object."""
    discount_factor: Optional[float | List[float]] = 0.6
    discount_factor_model: Optional[float | List[float]] = 0.8
    verbose: Optional[int | List[int]] = get_auto("verbose")
    tensorboard_log: Optional[str | List[str]] = get_auto("tensorboard_log")
    seed: Optional[int | List[int]] = get_auto("seed")
    learning_rate: Optional[float | List[float]] = 0.08
    hidden_size: Optional[int | List[int]] = 10

    class Config:
        extra = Extra.forbid


class ConfigIDHPLearn(BaseModel):
    """Allows defining parameters that can be passed to learn method."""
    total_timesteps: Optional[int] = 1_000
    log_interval: Optional[int] = 1
    progress_bar: Optional[bool] = False
    callback: Optional[list] = ["online", "tensorboard"]

    class Config:
        extra = Extra.forbid


class ConfigIDHP(BaseModel):
    name: Optional[Literal['IDHP']] = "IDHP"
    args: Optional[ConfigIDHPArgs] = ConfigIDHPArgs()
    kwargs: Optional[ConfigIDHPKwargs] = ConfigIDHPKwargs()
    sweep: Optional[ConfigIDHPKwargs] = ConfigIDHPKwargs()
    learn: Optional[ConfigIDHPLearn] = ConfigIDHPLearn()
    object: BaseAlgorithm = IDHP

    class Config:
        extra = Extra.forbid
