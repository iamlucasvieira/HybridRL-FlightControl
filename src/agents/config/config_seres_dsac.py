"""Module with Seres' DSAC configuration."""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

from agents.base_agent import BaseAgent
from agents.seres_dsac.seres_dsac_agent import DSAC


class ConfigSDSACArgs(BaseModel):
    """Arguments for Seres' DSAC object."""

    pass


class ConfigSDSACKwargs(BaseModel):
    """Keyword arguments for Seres' DSA object."""

    policy: Optional[str] = "MlpPolicy"


class ConfigSDSACSweep(BaseModel):
    """Allows defining parameters that can be swept over."""

    pass


class ConfigSDSAC(BaseModel):
    """Configuration of Seres' DSAC."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: Literal["SDSAC"] = "SDSAC"
    args: Optional[ConfigSDSACArgs] = ConfigSDSACArgs()
    kwargs: Optional[ConfigSDSACKwargs] = ConfigSDSACKwargs()
    sweep: Optional[ConfigSDSACSweep] = ConfigSDSACSweep()
    object: BaseAgent = DSAC
