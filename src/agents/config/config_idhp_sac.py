"""Module that defines the IDHP-SAC configuration."""

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict

from agents import IDHPSAC, BaseAgent
from agents.config.config_idhp import ConfigIDHPKwargs
from agents.config.config_sac import ConfigSACKwargs
from helpers.config_auto import get_auto


class ConfigIDHPSACArgs(BaseModel):
    """Arguments for IDHP-SAC object."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    env: Optional[str] = get_auto("env")


class ConfigIDHPSACKwargs(BaseModel):
    """Keyword arguments for IDHP-SAC object."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    policy_kwargs: Optional[dict | List[dict]] = None
    log_dir: Optional[str | List[str]] = get_auto("log_dir")
    save_dir: Optional[str | List[str]] = get_auto("save_dir")
    verbose: Optional[int | List[int]] = get_auto("verbose")
    seed: Optional[int | List[int]] = get_auto("seed")
    device: Optional[str | List[str]] = "cpu"
    _init_setup_model: Optional[bool | List[bool]] = True
    idhp_kwargs: Optional[ConfigIDHPKwargs | List[ConfigIDHPKwargs]] = (
        ConfigIDHPKwargs()
    )
    sac_kwargs: Optional[ConfigSACKwargs | List[ConfigSACKwargs]] = ConfigSACKwargs()
    idhp_actor_observation: Optional[str | List[str]] = "sac_attitude"


class ConfigIDHPSACLearn(BaseModel):
    """Allows defining parameters that can be passed to learn method."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    total_steps: Optional[int] = 1_000
    sac_steps: Optional[int] = 1_000
    idhp_steps: Optional[int] = 1_000
    sac_model: Optional[str] = None
    callback: Optional[list] = ["idhp_sac"]
    log_interval: Optional[int] = 100
    run_name: Optional[str] = get_auto("run_name")


class ConfigIDHPSAC(BaseModel):
    """Configuration of SAC."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: Literal["IDHP-SAC"] = "IDHP-SAC"
    args: Optional[ConfigIDHPSACArgs] = ConfigIDHPSACArgs()
    kwargs: Optional[ConfigIDHPSACKwargs] = ConfigIDHPSACKwargs()
    sweep: Optional[ConfigIDHPSACKwargs] = ConfigIDHPSACKwargs()
    learn: Optional[ConfigIDHPSACLearn] = ConfigIDHPSACLearn()
    object: BaseAgent = IDHPSAC
