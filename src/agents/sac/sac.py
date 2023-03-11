import gym
import torch
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import MaybeCallback
from typing import Union, Optional
import torch as th
from copy import deepcopy
from agents.sac.policy import SACPolicy
from agents.sac.buffer import ReplayBuffer, Transition
from helpers.torch import get_device, to_tensor, freeze, unfreeze


class SAC(BaseAlgorithm):
    """Implements the Soft Actor-Critic algorithm."""

    policy_aliases = {'default': SACPolicy}

    def __init__(self,
                 env: Union[gym.Env, str],
                 policy: Union[BasePolicy, str] = 'default',
                 learning_rate: float = 3e-4,
                 policy_kwargs: Optional[dict] = None,
                 tensorboard_log: Optional[str] = None,
                 verbose: int = 0,
                 seed: Optional[int] = None,
                 _init_setup_model: bool = True,
                 buffer_size: int = 1_000_000,
                 gradient_steps: int = 1,
                 batch_size: int = 256,
                 learning_starts: int = 100,
                 alpha: float = 0.2,
                 gamma: float = 0.99,
                 polyak: float = 0.995,
                 device: Optional[Union[th.device, str]] = None,
                 ):
        """Initialize the SAC algorithm.

        args:
            env: Environment.
            policy: Policy.
            learning_rate: Learning rate.
            policy_kwargs: Policy keyword arguments.
            tensorboard_log: Tensorboard log directory.
            verbose: Verbosity.
            seed: Random seed.
            _init_setup_model: Whether to initialize the model.
            buffer_size: Replay buffer size.
            batch_size: Batch size.
            learning_starts: Number of steps before learning starts.
            alpha: Entropy coefficient.
            gamma: Discount factor.
            polyak: Polyak averaging coefficient for updating target networks.
            device: Device to use.
        """
        if device is None:
            device = get_device()

        self.buffer_size = buffer_size
        self.gradient_steps = gradient_steps
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.alpha = alpha
        self.gamma = gamma
        self.polyak = polyak
        super(SAC, self).__init__(policy,
                                  env,
                                  learning_rate=learning_rate,
                                  policy_kwargs=policy_kwargs,
                                  tensorboard_log=tensorboard_log,
                                  verbose=verbose,
                                  seed=seed,
                                  device=device, )

        if _init_setup_model:
            self._setup_model()

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 100,
            tb_log_name: str = "run",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ) -> None:
        """Learn from the environment."""

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar, )

        callback.on_training_start(locals(), globals())
        callback.on_rollout_start()

        env = self.env.envs[0].env
        obs = env.reset()

        for step in range(total_timesteps):
            callback.on_step()
            self.num_timesteps += 1

            if step < self.learning_starts:
                action = env.action_space.sample()
            else:
                action = self.policy.get_action(obs)

            obs_tp1, reward, done, info = env.step(action)
            self.replay_buffer.push(Transition(obs=obs,
                                               action=action,
                                               reward=reward,
                                               obs_=obs_tp1,
                                               done=done))

            # If done, reset the environment
            if done:
                obs = env.reset()
            else:
                obs = obs_tp1

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

    def update_target_networks(self) -> None:
        """Update the target networks."""
        polyak = self.polyak
        with th.no_grad():
            for param, target_param in zip(self.policy.parameters(),
                                           self.target_policy.parameters()):
                new_target_param = polyak * target_param.data + (1 - polyak) * param.data
                target_param.data.copy_(new_target_param)


    def _setup_model(self) -> None:
        """Initialize the SAC policy and replay buffer"""
        self.set_random_seed(self.seed)
        self.policy = self.policy_class(self.observation_space,
                                        self.action_space,
                                        **self.policy_kwargs)
        self.target_policy = deepcopy(self.policy)
        self.replay_buffer = ReplayBuffer(self.buffer_size)

    def get_critic_loss(self, transition: Transition) -> th.Tensor:
        """Get the critic loss.

        args:
            transition: Transition tuple.
        """
        s_t, a_t = transition.obs, transition.action
        s_tp1 = transition.obs_
        r_t = transition.reward
        done = transition.done

        s_t, a_t, s_tp1, r_t, done = to_tensor(s_t, a_t, s_tp1, r_t, done)
        critic_1 = self.policy.critic_1(s_t, a_t)
        critic_2 = self.policy.critic_2(s_t, a_t)

        with th.no_grad():
            a_tp1, log_prob_tp1 = self.policy.actor(s_tp1)

            critic_1_tp1 = self.target_policy.critic_1(s_tp1, a_tp1)
            critic_2_tp1 = self.target_policy.critic_2(s_tp1, a_tp1)
            critic_tp1 = th.min(critic_1_tp1, critic_2_tp1)

            target = r_t + self.gamma * (1 - done) * (critic_tp1 - self.alpha * log_prob_tp1)

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
        s_t = to_tensor(transition.obs)

        a_t, log_prob = self.policy.actor(s_t)
        alpha = self.alpha

        critic_1 = self.policy.critic_1(s_t, a_t)
        critic_2 = self.policy.critic_2(s_t, a_t)
        critic = th.min(critic_1, critic_2)

        loss = (alpha * log_prob - critic).mean()
        return loss
