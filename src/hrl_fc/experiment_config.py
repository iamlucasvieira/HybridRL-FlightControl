"""Module that define configuration of algorithms."""
from pydantic import BaseModel, Field, Extra
from typing import Optional, List, Union, Literal

from agents.config import ConfigSB3SAC, ConfigIDHP, ConfigTD3, ConfigSDSAC, ConfigSAC
from envs.config.config_lti_env import ConfigLTIEnv
from envs.config.config_citation_env import ConfigCitationEnv
from envs.config.gym_env import ConfigGymEnv

class ConfigExperiment(BaseModel):
    """Class that defines the configuration of the sweep."""
    name: Optional[str]
    description: Optional[str]
    wandb: Optional[bool] = True
    n_learning: Optional[int] = 1
    evaluate: Optional[bool] = True
    verbose: Optional[int] = 1
    seed: Optional[int] = None
    save_model: Optional[bool] = True
    env: Union[ConfigLTIEnv, ConfigCitationEnv, ConfigGymEnv] = Field(discriminator='name', default=ConfigLTIEnv(name="LTI"))
    agent: Union[ConfigSB3SAC, ConfigIDHP, ConfigTD3, ConfigSDSAC, ConfigSAC] = Field(discriminator='name',
                                                                           default=ConfigSB3SAC(name="SB3SAC"))

    class Config:
        extra = Extra.forbid
