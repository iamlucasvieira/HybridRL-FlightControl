"""Module with SAC configuration."""

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict

from agents import SAC, BaseAgent
from helpers.config_auto import get_auto


class ConfigSACArgs(BaseModel):
    """Arguments for IDHP object."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    env: Optional[str] = get_auto("env")


class ConfigSACKwargs(BaseModel):
    """Keyword arguments for IDHP object."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    learning_rate: Optional[float | List[float]] = 3e-4
    policy_kwargs: Optional[dict | List[dict]] = None
    log_dir: Optional[str | List[str]] = get_auto("log_dir")
    save_dir: Optional[str | List[str]] = get_auto("save_dir")
    verbose: Optional[int | List[int]] = get_auto("verbose")
    seed: Optional[int | List[int]] = get_auto("seed")
    _init_setup_model: Optional[bool | List[bool]] = True
    buffer_size: Optional[int | List[int]] = 1_000_000
    gradient_steps: Optional[int | List[int]] = 1
    batch_size: Optional[int | List[int]] = 256
    learning_starts: Optional[int | List[int]] = 100
    entropy_coefficient: Optional[float | List[float]] = 0.2
    entropy_coefficient_update: Optional[bool | List[bool]] = True
    gamma: Optional[float | List[float]] = 0.99
    polyak: Optional[float | List[float]] = 0.995
    device: Optional[str | List[str]] = "cpu"


class ConfigSACLearn(BaseModel):
    """Allows defining parameters that can be passed to learn method."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    total_steps: Optional[int] = 1_000
    callback: Optional[list] = ["tensorboard", "sac"]
    log_interval: Optional[int] = 100
    run_name: Optional[str] = get_auto("run_name")


# Configuration of Agents
class ConfigSAC(BaseModel):
    """Configuration of SAC."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: Literal["SAC"] = "SAC"
    args: Optional[ConfigSACArgs] = ConfigSACArgs()
    kwargs: Optional[ConfigSACKwargs] = ConfigSACKwargs()
    sweep: Optional[ConfigSACKwargs] = ConfigSACKwargs()
    learn: Optional[ConfigSACLearn] = ConfigSACLearn()
    object: BaseAgent = SAC
