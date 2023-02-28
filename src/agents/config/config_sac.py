"""Module with SAC configuration."""
from pydantic import BaseModel, Extra
from typing import Optional, Literal
from stable_baselines3 import SAC
from stable_baselines3.common.base_class import BaseAlgorithm


class ConfigSACArgs(BaseModel):
    """Arguments for IDHP object."""
    pass


class ConfigSACKwargs(BaseModel):
    """Keyword arguments for IDHP object."""
    policy: Optional[str] = "default"


class ConfigSACSweep(BaseModel):
    """Allows defining parameters that can be swept over."""
    pass


# Configuration of Agents
class ConfigSAC(BaseModel):
    """Configuration of SAC."""
    name: Literal['SAC'] = "SAC"
    args: Optional[ConfigSACArgs] = ConfigSACArgs()
    kwargs: Optional[ConfigSACKwargs] = ConfigSACKwargs()
    sweep: Optional[ConfigSACSweep] = ConfigSACSweep()
    object: BaseAlgorithm = SAC

    class Config:
        extra = Extra.forbid
