"""Module with Seres' DSAC configuration."""
from pydantic import BaseModel, Extra
from typing import Optional, Literal
from agents.seres_dsac.seres_dsac_agent import DSAC
from stable_baselines3.common.base_class import BaseAlgorithm


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
    name: Literal['DSAC'] = "DSAC"
    args: Optional[ConfigSDSACArgs] = ConfigSDSACArgs()
    kwargs: Optional[ConfigSDSACKwargs] = ConfigSDSACKwargs()
    sweep: Optional[ConfigSDSACSweep] = ConfigSDSACSweep()
    object: BaseAlgorithm = DSAC

    class Config:
        extra = Extra.forbid
