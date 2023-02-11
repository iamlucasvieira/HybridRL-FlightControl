"""Module with DSAC configuration."""
from pydantic import BaseModel, validator, Field
from typing import Optional, List, Union, Literal


class ConfigSDSACSweep(BaseModel):
    pass


class ConfigSDSACBase(BaseModel):
    """Base configuration for SAC."""
    policy: Optional[str] = "MlpPolicy"
