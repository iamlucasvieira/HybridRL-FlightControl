"""Module that defines the IDHP-DSAC configuration."""
from typing import List, Literal, Optional

from pydantic import BaseModel, Extra

from agents import IDHPDSAC, BaseAgent
from helpers.config_auto import get_auto


class ConfigIDHPDSACArgs(BaseModel):
    """Arguments for IDHP-DSAC object."""

    env: Optional[str] = get_auto("env")

    class Config:
        extra = Extra.forbid


class ConfigIDHPDSACKwargs(BaseModel):
    """Keyword arguments for IDHP-DSAC object."""

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
    dsac_hidden_layers: Optional[List[int] | List[List[int]]] = None
    idhp_hidden_layers: Optional[List[int] | List[List[int]]] = None

    class Config:
        extra = Extra.forbid


class ConfigIDHPDSACLearn(BaseModel):
    """Allows defining parameters that can be passed to learn method."""

    total_steps: Optional[int] = 1_000
    dsac_steps: Optional[int] = 1_000
    idhp_steps: Optional[int] = 1_000
    dsac_model: Optional[str] = None
    callback: Optional[list] = ["idhp_sac"]
    log_interval: Optional[int] = 1
    run_name: Optional[str] = get_auto("run_name")

    class Config:
        extra = Extra.forbid


class ConfigIDHPDSAC(BaseModel):
    """Configuration of SAC."""

    name: Literal["IDHP-DSAC"] = "IDHP-DSAC"
    args: Optional[ConfigIDHPDSACArgs] = ConfigIDHPDSACArgs()
    kwargs: Optional[ConfigIDHPDSACKwargs] = ConfigIDHPDSACKwargs()
    sweep: Optional[ConfigIDHPDSACKwargs] = ConfigIDHPDSACKwargs()
    learn: Optional[ConfigIDHPDSACLearn] = ConfigIDHPDSACLearn()
    object: BaseAgent = IDHPDSAC

    class Config:
        extra = Extra.forbid
