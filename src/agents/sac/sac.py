"""Module that creates the SAC algorithm."""
from copy import deepcopy
from typing import Any, Optional, SupportsFloat, Type, Union

import gymnasium as gym
import numpy as np
import torch as th

from agents import BaseAgent
from agents.base_callback import ListCallback
from agents.buffer import ReplayBuffer, Transition
from agents.sac.policy import BasePolicy, SACPolicy
from envs import BaseEnv
from helpers.torch_helpers import freeze, to_tensor, unfreeze


class SAC(BaseAgent):
    """Implements the Soft Actor-Critic algorithm."""

    name = "SAC"

    def __init__(
        self,
        env: Union[gym.Env, Type[BaseEnv]],
        learning_rate: float = 3e-4,
        policy_kwargs: Optional[dict] = None,
        log_dir: Optional[str] = None,
        save_dir: Optional[str] = None,
        verbose: int = 1,
        seed: Optional[int] = None,
        _init_setup_model: bool = True,
        buffer_size: int = 1_000_000,
        gradient_steps: int = 1,
        batch_size: int = 256,
        learning_starts: int = 100,
        entropy_coefficient: float = 0.2,
        entropy_coefficient_update: bool = True,
        gamma: float = 0.99,
        polyak: float = 0.995,
        device: Optional[Union[th.device, str]] = None,
        policy: Optional[Type[BasePolicy]] = None,
    ):
        """Initialize the SAC algorithm.

        args:
            env: Environment.
            learning_rate: Learning rate.
            policy_kwargs: Policy keyword arguments.
            log_dir: Log directory.
            save_dir: Save directory.
            verbose: Verbosity.
            seed: Random seed.
            _init_setup_model: Whether to initialize the model.
            buffer_size: Replay buffer size.
            batch_size: Batch size.
            learning_starts: Number of steps before learning starts.
            entropy_coefficient: Entropy coefficient.
            entropy_coefficient_update: Whether to update the entropy coefficient.
            gamma: Discount factor.
            polyak: Polyak averaging coefficient for updating target networks.
            device: Device to use.
        """
        if policy is None:
            policy = SACPolicy
        if policy_kwargs is None:
            policy_kwargs = {}
        policy_kwargs["learning_rate"] = learning_rate

        self.buffer_size = buffer_size
        self.gradient_steps = gradient_steps
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.learning_rate = learning_rate
        self.entropy_coefficient = entropy_coefficient
        self.entropy_coefficient_update = entropy_coefficient_update
        self.gamma = gamma
        self.polyak = polyak

        self.log_ent_coef = None
        self.ent_coef_optimizer = None
        self.target_entropy = None

        super().__init__(
            policy,
            env,
            policy_kwargs=policy_kwargs,
            log_dir=log_dir,
            save_dir=save_dir,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

    def _learn(
        self,
        total_steps: int,
        callback: ListCallback,
        log_interval: int,
        **kwargs,
    ) -> None:
        """Learn from the environment."""
        env = self.env
        obs, _ = env.reset()
        episode_return = 0

        for step in range(total_steps):
            callback.on_step()
            self.num_steps += 1

            if step < self.learning_starts:
                action = env.action_space.sample()
            else:
                action = self.policy.get_action(obs)

            obs_tp1, reward, terminated, truncated, info = self.get_rollout(
                action, obs, callback
            )
            done = terminated or truncated
            episode_return += reward

            self.replay_buffer.push(
                Transition(
                    obs=obs, action=action, reward=reward, obs_=obs_tp1, done=done
                )
            )

            # If done, reset the environment
            if done:
                callback.on_episode_end(episode_return)
                obs, _ = env.reset()
                self._episode_num += 1
                episode_return = 0
            else:
                obs = obs_tp1

            if self.num_steps % log_interval == 0:
                self.dump_logs()

            if step >= self.learning_starts:
                for gradient_step in range(self.gradient_steps):
                    self.update()
                    self.update_target_networks()

    def update(self) -> None:
        """Update the policy."""
        buffer = self.replay_buffer.sample_buffer(self.batch_size)
        self.policy.critic_1.zero_grad()
        self.policy.critic_2.zero_grad()
        loss_critic = self.get_critic_loss(buffer)
        loss_critic.backward()
        self.policy.critic_1.optimizer.step()
        self.policy.critic_2.optimizer.step()

        # Freeze critic networks to avoid gradient computation during actor update
        freeze(self.policy.critic_1)
        freeze(self.policy.critic_2)

        self.policy.actor.zero_grad()
        loss_actor = self.get_actor_loss(buffer)
        loss_actor.backward()
        self.policy.actor.optimizer.step()

        # Unfreeze critic networks
        unfreeze(self.policy.critic_1)
        unfreeze(self.policy.critic_2)

        self._n_updates += 1

        if self.entropy_coefficient_update:
            self.update_entropy_coefficient(buffer)

        self.logger.record("train/n_updates", self._n_updates)
        self.logger.record("train/actor_loss", np.mean(loss_actor.mean().item()))
        self.logger.record("train/critic_loss", np.mean(loss_critic.mean().item()))

    def update_target_networks(self) -> None:
        """Update the target networks."""
        polyak = self.polyak
        with th.no_grad():
            for param, target_param in zip(
                self.policy.parameters(), self.target_policy.parameters()
            ):
                new_target_param = (
                    polyak * target_param.data + (1 - polyak) * param.data
                )
                target_param.data.copy_(new_target_param)

    def update_entropy_coefficient(self, buffer) -> None:
        """Update the entropy coefficient."""
        obs = to_tensor(buffer.obs, device=self.device)
        _, log_prob = self.policy.actor(obs)
        ent_coef = th.exp(self.log_ent_coef.detach())
        ent_coef_loss = -(
            self.log_ent_coef * (log_prob + self.target_entropy).detach()
        ).mean()

        self.ent_coef_optimizer.zero_grad()
        ent_coef_loss.backward()
        self.ent_coef_optimizer.step()

        self.entropy_coefficient = ent_coef.item()
        self.logger.record("train/ent_coef", ent_coef.mean().item())
        self.logger.record("train/ent_coef_loss", ent_coef_loss.mean().item())

    def setup_model(self) -> None:
        """Initialize the SAC policy and replay buffer"""
        self.target_policy = deepcopy(self.policy)
        self.replay_buffer = ReplayBuffer(self.buffer_size)

        if self.entropy_coefficient_update:
            self.log_ent_coef = th.log(
                th.ones(1, device=self.device) * self.entropy_coefficient
            ).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam(
                [self.log_ent_coef], lr=self.learning_rate
            )
            self.target_entropy = -np.prod(self.action_space.shape).astype(np.float32)

    def get_critic_loss(self, transition: Transition) -> th.Tensor:
        """Get the critic loss.

        args:
            transition: Transition tuple.
        """
        s_t, a_t = transition.obs, transition.action
        s_tp1 = transition.obs_
        r_t = transition.reward
        done = transition.done

        s_t, a_t, s_tp1, r_t, done = to_tensor(
            s_t, a_t, s_tp1, r_t, done, device=self.device
        )
        critic_1 = self.policy.critic_1(s_t, a_t)
        critic_2 = self.policy.critic_2(s_t, a_t)

        with th.no_grad():
            a_tp1, log_prob_tp1 = self.policy.actor(s_tp1)

            critic_1_target = self.target_policy.critic_1(s_tp1, a_tp1)
            critic_2_target = self.target_policy.critic_2(s_tp1, a_tp1)
            critic_target = th.min(critic_1_target, critic_2_target)
            alpha = self.entropy_coefficient
            target = r_t + self.gamma * (1 - done) * (
                critic_target - alpha * log_prob_tp1
            )

        loss_1 = ((critic_1 - target) ** 2).mean()
        loss_2 = ((critic_2 - target) ** 2).mean()
        # Adding the two loss is computationally more efficient as we backpropagate only once
        loss = loss_1 + loss_2
        return loss

    def get_actor_loss(self, transition: Transition) -> th.Tensor:
        """Get the actor loss.

        args:
            transition: Transition tuple.
        """
        s_t = to_tensor(transition.obs, device=self.device)

        a_t, log_prob = self.policy.actor(s_t)
        alpha = self.entropy_coefficient

        critic_1 = self.policy.critic_1(s_t, a_t)
        critic_2 = self.policy.critic_2(s_t, a_t)
        critic = th.min(critic_1, critic_2)

        loss = (alpha * log_prob - critic).mean()
        return loss

    def get_rollout(
        self,
        action: np.ndarray,
        obs: np.ndarray,
        callback: ListCallback,
        scale_action: bool = False,
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """Get the rollout."""
        return super().get_rollout(action, obs, callback, scale_action=scale_action)
