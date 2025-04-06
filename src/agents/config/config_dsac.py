"""Module that defines the DSAC configuration."""

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict

from agents import DSAC, BaseAgent
from helpers.config_auto import get_auto


class ConfigDSACArgs(BaseModel):
    """Arguments for DSAC object."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    env: Optional[str] = get_auto("env")


class ConfigDSACKwargs(BaseModel):
    """Keyword arguments for IDHP object."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    device: Optional[str | List[str]] = "cpu"
    verbose: Optional[int | List[int]] = get_auto("verbose")
    seed: Optional[int | List[int]] = get_auto("seed")
    log_dir: Optional[str | List[str]] = get_auto("log_dir")
    save_dir: Optional[str | List[str]] = get_auto("save_dir")
    policy_kwargs: Optional[dict | List[dict]] = None
    _init_setup_model: Optional[bool | List[bool]] = True
    learning_rate: Optional[float | List[float]] = 3e-4
    buffer_size: Optional[int | List[int]] = 1_000_000
    learning_starts: Optional[int | List[int]] = 10_000
    gradient_steps: Optional[int | List[int]] = 1
    batch_size: Optional[int | List[int]] = 256
    entropy_coefficient: Optional[float | List[float]] = 0.2
    entropy_coefficient_update: Optional[bool | List[bool]] = True
    gamma: Optional[float | List[float]] = 0.99
    polyak: Optional[float | List[float]] = 0.995
    num_quantiles: Optional[int | List[int]] = 32
    embedding_dim: Optional[int | List[int]] = 64
    hidden_layers: Optional[List[int] | List[List[int]]] = [256, 256]


class ConfigDSACLearn(BaseModel):
    """Allows defining parameters that can be passed to learn method."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    total_steps: Optional[int] = 1_000
    callback: Optional[list] = ["tensorboard", "sac"]
    log_interval: Optional[int] = 100
    run_name: Optional[str] = get_auto("run_name")


class ConfigDSAC(BaseModel):
    """Configuration of SAC."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: Literal["DSAC"] = "DSAC"
    args: Optional[ConfigDSACArgs] = ConfigDSACArgs()
    kwargs: Optional[ConfigDSACKwargs] = ConfigDSACKwargs()
    sweep: Optional[ConfigDSACKwargs] = ConfigDSACKwargs()
    learn: Optional[ConfigDSACLearn] = ConfigDSACLearn()
    object: BaseAgent = DSAC
