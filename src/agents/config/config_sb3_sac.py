"""Module with SAC configuration."""
from pydantic import BaseModel, Extra
from typing import Optional, Literal, List
from stable_baselines3 import SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from helpers.config_auto import get_auto


class ConfigSB3SACArgs(BaseModel):
    """Arguments for IDHP object."""
    policy: Optional[str] = "MlpPolicy"
    env: Optional[str] = get_auto("env")

    class Config:
        extra = Extra.forbid


class ConfigSB3SACKwargs(BaseModel):
    """Keyword arguments for IDHP object."""
    learning_rate: Optional[float | List[float]] = 3e-4
    buffer_size: Optional[int | List[int]] = 1_000_000  # 1e6
    learning_starts: Optional[int | List[int]] = 100
    batch_size: Optional[int | List[int]] = 256
    tau: Optional[float | List[float]] = 0.005
    gamma: Optional[float | List[float]] = 0.99
    train_freq: Optional[int | List[int]] = 1
    gradient_steps: Optional[int | List[int]] = 1
    # action_noise: Optional[ActionNoise] = None
    # replay_buffer_class: Optional[Type[ReplayBuffer]] = None
    # replay_buffer_kwargs: Optional[Dict[str, Any]] = None
    optimize_memory_usage: Optional[bool | List[bool]] = False
    ent_coef: Optional[str | List[str]] = "auto"
    target_update_interval: Optional[int | List[int]] = 1
    target_entropy: Optional[str | List[str]] = "auto"
    use_sde: Optional[bool | List[bool]] = False
    sde_sample_freq: Optional[int | List[int]] = -1
    use_sde_at_warmup: Optional[bool | List[bool]] = False
    tensorboard_log: Optional[str | List[str]] = get_auto("tensorboard_log")
    policy_kwargs: Optional[dict | List[dict]] = None
    verbose: Optional[int | List[int]] = get_auto("verbose")
    seed: Optional[int | List[int]] = get_auto("seed")
    device: Optional[str | List[str]] = "auto"
    _init_setup_model: Optional[bool | List[bool]] = True

    class Config:
        extra = Extra.forbid


class ConfigSB3SACLearn(BaseModel):
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
class ConfigSB3SAC(BaseModel):
    """Configuration of SAC."""
    name: Literal['SB3SAC'] = "SB3SAC"
    args: Optional[ConfigSB3SACArgs] = ConfigSB3SACArgs()
    kwargs: Optional[ConfigSB3SACKwargs] = ConfigSB3SACKwargs()
    sweep: Optional[ConfigSB3SACKwargs] = ConfigSB3SACKwargs()
    learn: Optional[ConfigSB3SACLearn] = ConfigSB3SACLearn()
    object: BaseAlgorithm = SAC

    class Config:
        extra = Extra.forbid
