"""Module that defines the DSAC algorithm."""

from typing import Optional, Type, Union

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn.functional as F

from agents.buffer import Transition
from agents.dsac.policy import DSACPolicy, check_dimensions, generate_quantiles
from agents.sac.sac import SAC
from envs import BaseEnv
from helpers.torch_helpers import freeze, to_tensor, unfreeze


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
        num_quantiles: int = 32,
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

        self.num_quantiles = num_quantiles

    def update(self) -> None:
        """Update the agent."""
        buffer = self.replay_buffer.sample_buffer(self.batch_size)

        self._n_updates += 1

        # Update the critic
        critic_loss = self.get_critic_loss(buffer)
        self.policy.z1.zero_grad()
        self.policy.z2.zero_grad()
        critic_loss.backward()
        self.policy.z1.optimizer.step()
        self.policy.z2.optimizer.step()

        # Freeze critic networks to avoid gradient computation during actor update
        freeze(self.policy.z1)
        freeze(self.policy.z2)

        self.policy.actor.zero_grad()
        loss_actor = self.get_actor_loss(buffer)
        loss_actor.backward()
        self.policy.actor.optimizer.step()

        # Unfreeze critic networks
        unfreeze(self.policy.z1)
        unfreeze(self.policy.z2)

        self.logger.record("train/critic_loss", np.mean(critic_loss.mean().item()))
        self.logger.record("train/actor_loss", np.mean(loss_actor.mean().item()))

    def get_critic_loss(self, transition: Transition) -> th.Tensor:
        """Get the critic loss."""
        s_t, a_t = transition.obs, transition.action
        s_tp1 = transition.obs_
        r_t = transition.reward
        done = transition.done

        s_t, a_t, s_tp1, r_t, done = to_tensor(
            s_t, a_t, s_tp1, r_t, done, device=self.device
        )

        batch_size = len(s_t)
        with th.no_grad():
            a_tp1, log_prob_tp1 = self.target_policy.actor(s_tp1)

            tau_i = generate_quantiles(
                batch_size, self.num_quantiles, device=self.device
            )
            tau_j = generate_quantiles(
                batch_size, self.num_quantiles, device=self.device
            )

            z1_target = self.target_policy.z1(s_t, a_t, tau_i)
            z2_target = self.target_policy.z2(s_t, a_t, tau_i)
            z_target = th.min(z1_target, z2_target)

            # Target
            alpha = self.entropy_coefficient

            target = r_t.view(-1, 1) + self.gamma * (1 - done.view(-1, 1)) * (
                z_target - alpha * log_prob_tp1.unsqueeze(-1)
            )

        z1 = self.policy.z1(s_t, a_t, tau_j)
        z2 = self.policy.z2(s_t, a_t, tau_j)
        loss_1 = self.huber_quantile_loss(target, z1, tau_j)
        loss_2 = self.huber_quantile_loss(target, z2, tau_j)
        loss = loss_1 + loss_2
        return loss

    def get_actor_loss(self, transition: Transition) -> th.Tensor:
        """Get the actor loss."""
        s_t, a_t = transition.obs, transition.action
        s_tp1 = transition.obs_
        r_t = transition.reward
        done = transition.done

        s_t, a_t, s_tp1, r_t, done = to_tensor(
            s_t, a_t, s_tp1, r_t, done, device=self.device
        )

        batch_size = len(s_t)

        a_tp1, log_prob_tp1 = self.policy.actor(s_tp1)

        tau_i = generate_quantiles(batch_size, self.num_quantiles, device=self.device)

        z1 = self.policy.z1(s_t, a_t, tau_i)
        z2 = self.policy.z2(s_t, a_t, tau_i)

        # Compute the action-value function
        q1 = z1.mean(dim=1)
        q2 = z2.mean(dim=1)
        q = th.min(q1, q2)

        # Compute the actor loss
        alpha = self.entropy_coefficient
        loss = (alpha * log_prob_tp1 - q).mean()

        return loss

    @staticmethod
    def huber_quantile_loss(
        z_target: th.Tensor, z: th.Tensor, quantiles: th.Tensor, threshold: float = 1.0
    ) -> th.Tensor:
        """Returns the Huber loss.

        Args:
            z_target: The target quantile values.
            z: The predicted quantile values.
            quantiles: The quantiles.
            threshold: The threshold for the Huber loss.

        Dimensions:
            z_target: (batch_size, num_quantiles)
            z: (batch_size, num_quantiles)
            quantiles: (batch_size, num_quantiles, 1)
        """
        # Check dimensions
        check_dimensions(z_target, 2, name="Target")
        check_dimensions(z, 2, name="Predicted")
        check_dimensions(quantiles, 3, name="Quantiles")

        # Get Temporal Difference Error
        td_error = z_target - z

        # Get Huber Loss
        huber_loss = F.huber_loss(
            z, z_target, reduction="none", delta=threshold
        )  # (batch_size, num_quantiles)

        # Get Quantile Huber Loss
        quantile_huber_loss = (
            (quantiles.squeeze(-1) - (td_error.detach() < 0).float())
            * huber_loss
            / threshold
        )  # (batch_size, num_quantiles)

        return quantile_huber_loss.sum(dim=1).mean()
