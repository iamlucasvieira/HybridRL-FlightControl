"""Module with TD3 configuration."""
from pydantic import BaseModel, Extra
from typing import Optional, Literal
from stable_baselines3 import TD3
from stable_baselines3.common.base_class import BaseAlgorithm


class ConfigTD3Args(BaseModel):
    """Arguments for IDHP object."""
    pass


class ConfigTD3Kwargs(BaseModel):
    """Keyword arguments for IDHP object."""
    policy: Optional[str] = "default"


class ConfigTD3Sweep(BaseModel):
    """Allows defining parameters that can be swept over."""
    pass


class ConfigTD3(BaseModel):
    """Configuration of TD3."""
    name: Literal['TD3'] = "TD3"
    args: Optional[ConfigTD3Args] = ConfigTD3Args()
    kwargs: Optional[ConfigTD3Kwargs] = ConfigTD3Kwargs()
    sweep: Optional[ConfigTD3Sweep] = ConfigTD3Sweep()
    object: BaseAlgorithm = TD3

    class Config:
        extra = Extra.forbid
