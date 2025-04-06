"""Module that defines the IDHP-DSAC configuration."""

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict

from agents import IDHPDSAC, BaseAgent
from agents.config.config_dsac import ConfigDSACKwargs
from agents.config.config_idhp import ConfigIDHPKwargs
from helpers.config_auto import get_auto


class ConfigIDHPDSACArgs(BaseModel):
    """Arguments for IDHP-DSAC object."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    env: Optional[str] = get_auto("env")


class ConfigIDHPDSACKwargs(BaseModel):
    """Keyword arguments for IDHP-DSAC object."""

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
    dsac_kwargs: Optional[ConfigDSACKwargs | List[ConfigDSACKwargs]] = (
        ConfigDSACKwargs()
    )
    idhp_actor_observation: Optional[str | List[str]] = "sac_attitude"


class ConfigIDHPDSACLearn(BaseModel):
    """Allows defining parameters that can be passed to learn method."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    total_steps: Optional[int] = 1_000
    dsac_steps: Optional[int] = 1_000
    idhp_steps: Optional[int] = 1_000
    sac_model: Optional[str] = None
    callback: Optional[list] = ["idhp_sac"]
    log_interval: Optional[int] = 100
    run_name: Optional[str] = get_auto("run_name")


class ConfigIDHPDSAC(BaseModel):
    """Configuration of SAC."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: Literal["IDHP-DSAC"] = "IDHP-DSAC"
    args: Optional[ConfigIDHPDSACArgs] = ConfigIDHPDSACArgs()
    kwargs: Optional[ConfigIDHPDSACKwargs] = ConfigIDHPDSACKwargs()
    sweep: Optional[ConfigIDHPDSACKwargs] = ConfigIDHPDSACKwargs()
    learn: Optional[ConfigIDHPDSACLearn] = ConfigIDHPDSACLearn()
    object: BaseAgent = IDHPDSAC
