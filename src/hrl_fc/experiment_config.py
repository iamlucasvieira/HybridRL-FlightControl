"""Module that define configuration of algorithms."""
from pydantic import BaseModel, Field, Extra
from typing import Optional, List, Union, Literal
import gym

from agents.config import ConfigSAC, ConfigIDHP, ConfigTD3, ConfigSDSAC

from envs.lti_citation.config_lti_env import ConfigLTIBase, ConfigLTISweep
from envs.lti_citation.aircraft_environment import AircraftEnv


# Configuration of Environments
class ConfigLTIEnv(BaseModel):
    """Symmetric derivatives."""
    name: Literal['LTI'] = "LTI"
    config: Optional[ConfigLTIBase] = ConfigLTIBase()
    sweep: Optional[ConfigLTISweep] = ConfigLTISweep()
    object: gym.Env = AircraftEnv

    class Config:
        extra = Extra.forbid


class Config6DOF(BaseModel):
    name: Literal['6DOF']


class ConfigAgent(BaseModel):
    """Configuration for RL algorithm."""
    __root__: Union[ConfigSAC, ConfigIDHP, ConfigTD3, ConfigSDSAC] = Field(discriminator='name',
                                                                           default=ConfigSAC(name="SAC"))


class ConfigExperiment(BaseModel):
    """Class that defines the configuration of the sweep."""
    name: Optional[str]
    wandb: Optional[bool] = True
    n_learning: Optional[int] = 1
    evaluate: Optional[bool] = True
    verbose: Optional[int] = 1
    seed: Optional[int] = None
    env: Union[ConfigLTIEnv, Config6DOF] = Field(discriminator='name', default=ConfigLTIEnv(name="LTI"))
    agent: Union[List[ConfigAgent], ConfigAgent] = ConfigAgent()

    class Config:
        extra = Extra.forbid
