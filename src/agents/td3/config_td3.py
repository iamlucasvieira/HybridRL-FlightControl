"""Module with TD3 configuration."""
from pydantic import BaseModel, validator, Field
from typing import Optional, List, Union, Literal


class ConfigTD3Sweep(BaseModel):
    pass


class ConfigTD3Base(BaseModel):
    """Base configuration for SAC."""
    policy: Optional[str] = "MlpPolicy"
