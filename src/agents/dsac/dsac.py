"""Module that defines the DSAC algorithm."""

from typing import Optional, Type, Union

import gymnasium as gym
import torch as th

from agents.buffer import Transition
from agents.dsac.policy import DSACPolicy
from agents.sac.sac import SAC
from envs import BaseEnv


class DSAC(SAC):
    """Implements the DSAC agent."""

    def __init__(
        self,
        env: Union[Type[BaseEnv], gym.Env],
        verbose: int = 0,
        seed: Optional[int] = None,
        log_dir: Optional[str] = None,
        save_dir: Optional[str] = None,
        policy_kwargs: Optional[dict] = None,
        _init_setup_model: bool = True,
        learning_rate: float = 3e-4,
        buffer_size: int = 100_000,
        learning_starts: int = 1_000,
        gradient_steps: int = 1,
        batch_size: int = 256,
        entropy_coefficient: float = 0.2,
        entropy_coefficient_update: bool = True,
        gamma: float = 0.99,
        polyak: float = 0.995,
        device: Optional[Union[th.device, str]] = None,
    ) -> None:
        """Initialize DSAC agent."""
        super().__init__(
            env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            log_dir=log_dir,
            save_dir=save_dir,
            verbose=verbose,
            seed=seed,
            _init_setup_model=_init_setup_model,
            buffer_size=buffer_size,
            gradient_steps=gradient_steps,
            batch_size=batch_size,
            learning_starts=learning_starts,
            entropy_coefficient=entropy_coefficient,
            entropy_coefficient_update=entropy_coefficient_update,
            gamma=gamma,
            polyak=polyak,
            device=device,
            policy=DSACPolicy,
        )

    def update(self) -> None:
        """Update the agent."""
        buffer = self.replay_buffer.sample_buffer(self.batch_size)
        self.policy.z1.zero_grad()
        self.policy.z2.zero_grad()
        loss_critic = self.get_critic_loss(buffer)

    def get_critic_loss(self, transition: Transition) -> th.Tensor:
        pass

    def get_actor_loss(self, transition: Transition) -> th.Tensor:
        pass
