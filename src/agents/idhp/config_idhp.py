"""Module with IHDP configuration."""
from pydantic import BaseModel
from typing import Optional


class ConfigIDHPSweep(BaseModel):
    pass


class ConfigIDHPBase(BaseModel):
    """Base configuration for SAC."""
    policy: Optional[str] = "default"
    gamma: Optional[float] = 0.99
