"""Module that defines the DSAC algorithm."""

from typing import Union, Optional

import gymnasium as gym

from agents import BaseAgent
from agents.dsac.policy import DSACPolicy
from envs import BaseEnv


class DSAC(BaseAgent):
    """DSAC agent."""

    def __init__(self,
                 env: Union[BaseEnv, gym.Env],
                 device: Optional[str] = None,
                 verbose: int = 0,
                 seed: Optional[int] = None,
                 log_dir: Optional[str] = None,
                 save_dir: Optional[str] = None,
                 policy_kwargs: Optional[dict] = None,
                 _init_setup_model: bool = True) -> None:
        """Initialize DSAC agent.

        Args:
            env: Environment.
            device: Device to use.
            verbose: Verbosity level.
            seed: Seed.
            log_dir: Log directory.
            save_dir: Save directory.
            policy_kwargs: Policy keyword arguments.
            _init_setup_model: Whether to initialize the model.
        """
        super(DSAC, self).__init__(DSACPolicy, env,
                                   device=device,
                                   verbose=verbose,
                                   seed=seed,
                                   log_dir=log_dir,
                                   save_dir=save_dir,
                                   policy_kwargs=policy_kwargs,
                                   _init_setup_model=_init_setup_model)