"""Module that define configuration of algorithms."""
from pydantic import BaseModel, Field, Extra
from typing import Optional, List, Union, Literal

from agents.config import ConfigSAC, ConfigIDHP, ConfigTD3, ConfigSDSAC
from envs.config.config_lti_env import ConfigLTIEnv
from envs.config.config_citation_env import ConfigCitationEnv


class ConfigExperiment(BaseModel):
    """Class that defines the configuration of the sweep."""
    name: Optional[str]
    wandb: Optional[bool] = True
    n_learning: Optional[int] = 1
    evaluate: Optional[bool] = True
    verbose: Optional[int] = 1
    seed: Optional[int] = None
    env: Union[ConfigLTIEnv, ConfigCitationEnv] = Field(discriminator='name', default=ConfigLTIEnv(name="LTI"))
    agent: Union[ConfigSAC, ConfigIDHP, ConfigTD3, ConfigSDSAC] = Field(discriminator='name',
                                                                        default=ConfigSAC(name="SAC"))

    class Config:
        extra = Extra.forbid
