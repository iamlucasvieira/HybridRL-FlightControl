"""Module with SAC configuration."""
from pydantic import BaseModel, validator, Field
from typing import Optional, List, Union, Literal


class ConfigSACSweep(BaseModel):
    pass


class ConfigSACBase(BaseModel):
    """Base configuration for SAC."""
    policy: Optional[str] = "MlpPolicy"
