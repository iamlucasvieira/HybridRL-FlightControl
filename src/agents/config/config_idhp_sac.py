from typing import Optional, Literal, List

from pydantic import BaseModel, Extra
from stable_baselines3.common.base_class import BaseAlgorithm

from agents import IDHPSAC
from helpers.config_auto import get_auto


class ConfigIDHPSACArgs(BaseModel):
    """Arguments for IDHP-SAC object."""
    policy: Optional[str] = "default"
    env: Optional[str] = get_auto("env")

    class Config:
        extra = Extra.forbid


class ConfigIDHPSACKwargs(BaseModel):
    """Keyword arguments for IDHP-SAC object."""
    learning_rate: Optional[float | List[float]] = 3e-4
    learning_starts: Optional[int | List[int]] = 100
    buffer_size: Optional[int | List[int]] = 1_000_000
    batch_size: Optional[int | List[int]] = 256
    policy_kwargs: Optional[dict | List[dict]] = None
    tensorboard_log: Optional[str | List[str]] = get_auto("tensorboard_log")
    verbose: Optional[int | List[int]] = get_auto("verbose")
    seed: Optional[int | List[int]] = get_auto("seed")
    device: Optional[str | List[str]] = None
    _init_setup_model: Optional[bool | List[bool]] = True
    sac_hidden_layers: Optional[List[int] | List[List[int]]] = None
    idhp_hidden_layers: Optional[List[int] | List[List[int]]] = None

    class Config:
        extra = Extra.forbid


class ConfigIDHPSACLearn(BaseModel):
    """Allows defining parameters that can be passed to learn method."""
    sac_timesteps: Optional[int] = 1_000
    idhp_timesteps: Optional[int] = 1_000
    sac_model: Optional[str] = None
    callback: Optional[list] = ["tensorboard"]
    log_interval: Optional[int] = 1
    tb_log_name: Optional[str] = get_auto("tb_log_name")
    reset_num_timesteps: Optional[bool] = True
    progress_bar: Optional[bool] = False


class ConfigIDHPSAC(BaseModel):
    """Configuration of SAC."""
    name: Literal['IDHP-SAC'] = "IDHP-SAC"
    args: Optional[ConfigIDHPSACArgs] = ConfigIDHPSACArgs()
    kwargs: Optional[ConfigIDHPSACKwargs] = ConfigIDHPSACKwargs()
    sweep: Optional[ConfigIDHPSACKwargs] = ConfigIDHPSACKwargs()
    learn: Optional[ConfigIDHPSACLearn] = ConfigIDHPSACLearn()
    object: BaseAlgorithm = IDHPSAC

    class Config:
        extra = Extra.forbid
