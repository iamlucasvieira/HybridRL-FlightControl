"""Module that defines the IDHP-SAC configuration."""
from typing import List, Literal, Optional

from pydantic import BaseModel, Extra

from agents import IDHPSAC, BaseAgent
from helpers.config_auto import get_auto


class ConfigIDHPSACArgs(BaseModel):
    """Arguments for IDHP-SAC object."""

    env: Optional[str] = get_auto("env")

    class Config:
        extra = Extra.forbid


class ConfigIDHPSACKwargs(BaseModel):
    """Keyword arguments for IDHP-SAC object."""

    learning_rate: Optional[float | List[float]] = 3e-4
    learning_starts: Optional[int | List[int]] = 100
    buffer_size: Optional[int | List[int]] = 1_000_000
    batch_size: Optional[int | List[int]] = 256
    policy_kwargs: Optional[dict | List[dict]] = None
    log_dir: Optional[str | List[str]] = get_auto("log_dir")
    save_dir: Optional[str | List[str]] = get_auto("save_dir")
    verbose: Optional[int | List[int]] = get_auto("verbose")
    seed: Optional[int | List[int]] = get_auto("seed")
    device: Optional[str | List[str]] = "cpu"
    _init_setup_model: Optional[bool | List[bool]] = True
    sac_hidden_layers: Optional[List[int] | List[List[int]]] = None
    idhp_hidden_layers: Optional[List[int] | List[List[int]]] = None

    class Config:
        extra = Extra.forbid


class ConfigIDHPSACLearn(BaseModel):
    """Allows defining parameters that can be passed to learn method."""

    total_steps: Optional[int] = 1_000
    sac_steps: Optional[int] = 1_000
    idhp_steps: Optional[int] = 1_000
    sac_model: Optional[str] = None
    callback: Optional[list] = ["tensorboard"]
    log_interval: Optional[int] = 1
    run_name: Optional[str] = get_auto("run_name")

    class Config:
        extra = Extra.forbid


class ConfigIDHPSAC(BaseModel):
    """Configuration of SAC."""

    name: Literal["IDHP-SAC"] = "IDHP-SAC"
    args: Optional[ConfigIDHPSACArgs] = ConfigIDHPSACArgs()
    kwargs: Optional[ConfigIDHPSACKwargs] = ConfigIDHPSACKwargs()
    sweep: Optional[ConfigIDHPSACKwargs] = ConfigIDHPSACKwargs()
    learn: Optional[ConfigIDHPSACLearn] = ConfigIDHPSACLearn()
    object: BaseAgent = IDHPSAC

    class Config:
        extra = Extra.forbid
