"""Module that define configuration of algorithms."""
from typing import Optional, Union

from pydantic import BaseModel, Extra, Field

from agents.config import (
    ConfigDSAC,
    ConfigIDHP,
    ConfigIDHPDSAC,
    ConfigIDHPSAC,
    ConfigSAC,
    ConfigSDSAC,
)
from envs.config.config_citation_env import ConfigCitationEnv
from envs.config.config_lti_env import ConfigLTIEnv
from envs.config.gym_env import ConfigGymEnv


class ConfigExperiment(BaseModel):
    """Class that defines the configuration of the sweep."""

    name: Optional[str]
    description: Optional[str]
    wandb: Optional[bool] = True
    n_learning: Optional[int] = 1
    evaluate: Optional[bool | int] = 1
    verbose: Optional[int] = 1
    seed: Optional[int] = None
    save_model: Optional[bool] = True
    env: Union[ConfigLTIEnv, ConfigCitationEnv, ConfigGymEnv] = Field(
        discriminator="name", default=ConfigLTIEnv(name="LTI")
    )
    agent: Union[
        ConfigIDHP, ConfigSDSAC, ConfigSAC, ConfigIDHPSAC, ConfigDSAC, ConfigIDHPDSAC
    ] = Field(discriminator="name", default=ConfigSAC(name="SAC"))

    class Config:
        extra = Extra.forbid
