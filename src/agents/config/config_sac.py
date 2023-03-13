"""Module with SAC configuration."""
from pydantic import BaseModel, Extra
from typing import Optional, Literal, List
from agents.sac.sac import SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from helpers.config_auto import get_auto


class ConfigSACArgs(BaseModel):
    """Arguments for IDHP object."""
    env: Optional[str] = get_auto("env")

    class Config:
        extra = Extra.forbid


class ConfigSACKwargs(BaseModel):
    """Keyword arguments for IDHP object."""
    policy: Optional[str | List[str]] = 'default'
    learning_rate: Optional[float | List[float]] = 3e-4
    policy_kwargs: Optional[dict | List[dict]] = None
    tensorboard_log: Optional[str | List[str]] = get_auto("tensorboard_log")
    verbose: Optional[int | List[int]] = get_auto("verbose")
    seed: Optional[int | List[int]] = None
    _init_setup_model: Optional[bool | List[bool]] = True
    buffer_size: Optional[int | List[int]] = 1_000_000
    gradient_steps: Optional[int | List[int]] = 1
    batch_size: Optional[int | List[int]] = 256
    learning_starts: Optional[int | List[int]] = 100
    entropy_coefficient: Optional[float | List[float]] = 0.2
    entropy_coefficient_update: Optional[bool | List[bool]] = True
    gamma: Optional[float | List[float]] = 0.99
    polyak: Optional[float | List[float]] = 0.995
    device: Optional[str | List[str]] = None

    class Config:
        extra = Extra.forbid


class ConfigSACLearn(BaseModel):
    """Allows defining parameters that can be passed to learn method."""
    total_timesteps: Optional[int] = 1_000
    callback: Optional[list] = ["tensorboard"]
    log_interval: Optional[int] = 1
    tb_log_name: Optional[str] = get_auto("tb_log_name")
    reset_num_timesteps: Optional[bool] = True
    progress_bar: Optional[bool] = False

    class Config:
        extra = Extra.forbid


# Configuration of Agents
class ConfigSAC(BaseModel):
    """Configuration of SAC."""
    name: Literal['SAC'] = "SAC"
    args: Optional[ConfigSACArgs] = ConfigSACArgs()
    kwargs: Optional[ConfigSACKwargs] = ConfigSACKwargs()
    sweep: Optional[ConfigSACKwargs] = ConfigSACKwargs()
    learn: Optional[ConfigSACLearn] = ConfigSACLearn()
    object: BaseAlgorithm = SAC

    class Config:
        extra = Extra.forbid
