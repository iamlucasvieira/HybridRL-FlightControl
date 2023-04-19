"""Module with IHDP configuration."""
from typing import List, Literal, Optional

from pydantic import BaseModel, Extra, validator

from agents import IDHP, BaseAgent
from agents.idhp.excitation import AVAILABLE_EXCITATION_FUNCTIONS
from helpers.config_auto import get_auto


class ConfigIDHPArgs(BaseModel):
    """Arguments for IDHP object."""

    env: Optional[str] = get_auto("env")

    class Config:
        extra = Extra.forbid


class ConfigIDHPKwargs(BaseModel):
    """Keyword arguments for IDHP object."""

    discount_factor: Optional[float | List[float]] = 0.6
    discount_factor_model: Optional[float | List[float]] = 0.8
    verbose: Optional[int | List[int]] = get_auto("verbose")
    log_dir: Optional[str | List[str]] = get_auto("log_dir")
    save_dir: Optional[str | List[str]] = get_auto("save_dir")
    seed: Optional[int | List[int]] = get_auto("seed")
    hidden_size: Optional[int | List[int]] = 10
    device: Optional[str | List[str]] = "cpu"
    excitation: Optional[str | List[str]] = None
    lr_a_low: Optional[float | List[float]] = 0.005
    lr_a_high: Optional[float | List[float]] = 0.08
    lr_c_low: Optional[float | List[float]] = 0.0005
    lr_c_high: Optional[float | List[float]] = 0.005
    lr_threshold: Optional[float | List[float]] = 0.001
    t_warmup: Optional[int | List[int]] = 100

    class Config:
        extra = Extra.forbid

    @validator("excitation")
    def excitation_validator(cls, v):
        if v is not None and v not in AVAILABLE_EXCITATION_FUNCTIONS:
            raise ValueError(
                f"Excitation must be in {list(AVAILABLE_EXCITATION_FUNCTIONS.keys())}, not '{v}'"
            )
        return v


class ConfigIDHPLearn(BaseModel):
    """Allows defining parameters that can be passed to learn method."""

    total_steps: Optional[int] = 1_000
    callback: Optional[list] = ["online", "tensorboard", "idhp"]
    log_interval: Optional[int] = 1
    run_name: Optional[str] = get_auto("run_name")

    class Config:
        extra = Extra.forbid


class ConfigIDHP(BaseModel):
    name: Optional[Literal["IDHP"]] = "IDHP"
    args: Optional[ConfigIDHPArgs] = ConfigIDHPArgs()
    kwargs: Optional[ConfigIDHPKwargs] = ConfigIDHPKwargs()
    sweep: Optional[ConfigIDHPKwargs] = ConfigIDHPKwargs()
    learn: Optional[ConfigIDHPLearn] = ConfigIDHPLearn()
    object: BaseAgent = IDHP

    class Config:
        extra = Extra.forbid
