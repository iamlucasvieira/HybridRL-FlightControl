"""Module that define configuration of algorithms."""
from pydantic import BaseModel, validator, Field, Extra
from typing import Optional, List, Union, Literal
import gym
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm

from envs.lti_citation.config_lti_env import ConfigLTIBase, ConfigLTISweep
from envs.lti_citation.aircraft_environment import AircraftEnv
from agents.sac.config_sac import ConfigSACBase, ConfigSACSweep
from agents.td3.config_td3 import ConfigTD3Base, ConfigTD3Sweep
from agents.seres_dsac.config_seres_dsac import ConfigSDSACBase, ConfigSDSACSweep
from agents.seres_dsac.seres_dsac_agent import DSAC
from agents.idhp.config_idhp import ConfigIDHPBase, ConfigIDHPSweep
from agents.idhp import IDHP


# Configuration of Environments
class ConfigLTIEnv(BaseModel):
    """Symmetric derivatives."""
    name: Literal['LTI'] = "LTI"
    config: Optional[ConfigLTIBase] = ConfigLTIBase()
    sweep: Optional[ConfigLTISweep] = ConfigLTISweep()
    object: gym.Env = AircraftEnv

    class Config:
        extra = Extra.forbid


# Configuration of Agents
class ConfigSAC(BaseModel):
    """Configuration of SAC."""
    name: Literal['SAC'] = "SAC"
    config: Optional[ConfigSACBase] = ConfigSACBase()
    sweep: Optional[ConfigSACSweep] = ConfigSACSweep()
    object: BaseAlgorithm = SAC

    class Config:
        extra = Extra.forbid


class ConfigTD3(BaseModel):
    """Configuration of TD3."""
    name: Literal['TD3'] = "TD3"
    config: Optional[ConfigSACBase] = ConfigTD3Base()
    sweep: Optional[ConfigSACSweep] = ConfigTD3Sweep()
    object: BaseAlgorithm = TD3

    class Config:
        extra = Extra.forbid


class ConfigDSAC(BaseModel):
    """Configuration of DSAC."""
    name: Literal['DSAC'] = "DSAC"
    config: Optional[ConfigSDSACBase] = ConfigSDSACBase()
    sweep: Optional[ConfigSDSACSweep] = ConfigSDSACSweep()
    object: BaseAlgorithm = DSAC

    class Config:
        extra = Extra.forbid


class ConfigIDHP(BaseModel):
    name: Optional[Literal['IDHP']] = "IDHP"
    config: Optional[ConfigIDHPBase] = ConfigIDHPBase()
    sweep: Optional[ConfigIDHPSweep] = ConfigIDHPSweep()
    object: BaseAlgorithm = IDHP


class Config6DOF(BaseModel):
    name: Literal['6DOF']


class ConfigAgent(BaseModel):
    """Configuration for RL algorithm."""
    __root__: Union[ConfigSAC, ConfigIDHP, ConfigTD3, ConfigDSAC] = Field(discriminator='name',
                                                                          default=ConfigSAC(name="SAC"))


class ConfigExperiment(BaseModel):
    """Class that defines the configuration of the sweep."""
    name: Optional[str]
    offline: Optional[bool] = False
    n_learning: Optional[int] = 1
    learning_steps: Optional[int] = 1_000  # Number of total learning steps
    env: Union[ConfigLTIEnv, Config6DOF] = Field(discriminator='name', default=ConfigLTIEnv(name="LTI"))
    agent: Union[List[ConfigAgent], ConfigAgent] = ConfigAgent()
    evaluate: Optional[bool] = True
    log_interval: Optional[int] = 1
    verbose: Optional[int] = 1
    seed: Optional[int] = None

    class Config:
        extra = Extra.forbid
