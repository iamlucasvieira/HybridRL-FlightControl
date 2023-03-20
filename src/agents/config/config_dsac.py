"""Module that defines the DSAC configuration."""
from typing import List
from typing import Literal
from typing import Optional

from pydantic import BaseModel
from pydantic import Extra

from agents import DSAC
from agents import BaseAgent
from helpers.config_auto import get_auto


class ConfigDSACArgs(BaseModel):
    """Arguments for DSAC object."""

    env: Optional[str] = get_auto("env")

    class Config:
        extra = Extra.forbid


class ConfigDSACKwargs(BaseModel):
    """Keyword arguments for IDHP object."""

    device: Optional[str | List[str]] = None
    verbose: Optional[int | List[int]] = get_auto("verbose")
    seed: Optional[int | List[int]] = get_auto("seed")
    log_dir: Optional[str | List[str]] = get_auto("log_dir")
    save_dir: Optional[str | List[str]] = get_auto("save_dir")
    policy_kwargs: Optional[dict | List[dict]] = None
    _init_setup_model: Optional[bool | List[bool]] = True

    class Config:
        extra = Extra.forbid


class ConfigDSACLearn(BaseModel):
    """Allows defining parameters that can be passed to learn method."""

    total_steps: Optional[int] = 1_000
    callback: Optional[list] = ["tensorboard"]
    log_interval: Optional[int] = 1
    run_name: Optional[str] = get_auto("run_name")

    class Config:
        extra = Extra.forbid


class ConfigDSAC(BaseModel):
    """Configuration of SAC."""

    name: Literal["SAC"] = "SAC"
    args: Optional[ConfigDSACArgs] = ConfigDSACArgs()
    kwargs: Optional[ConfigDSACKwargs] = ConfigDSACKwargs()
    sweep: Optional[ConfigDSACKwargs] = ConfigDSACKwargs()
    learn: Optional[ConfigDSACLearn] = ConfigDSACLearn()
    object: BaseAgent = DSAC

    class Config:
        extra = Extra.forbid
