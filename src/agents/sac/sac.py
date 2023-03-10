import gym
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from typing import Union, Optional
import torch as th
from copy import deepcopy
from agents.sac.policy import SACPolicy
from agents.sac.buffer import ReplayBuffer, Transition
from helpers.torch import get_device


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
                 batch_size: int = 256,
                 learning_starts: int = 100,
                 alpha: float = 0.2,
                 gamma: float = 0.99,
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
            device: Device to use.
        """
        if device is None:
            device = get_device()

        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.alpha = alpha
        self.gamma = gamma
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

    def learn(self, total_timesteps: int,
              callback=None,
              log_interval: int = 4,
              tb_log_name: str = "SAC",
              reset_num_timesteps: bool = True) -> None:
        """Learn from the environment."""

        callback = self._init_callback(callback)

        callback.on_training_start(locals(), globals())
        callback.on_rollout_start()

        env = self.env
        obs = env.reset()

        for step in range(total_timesteps):
            callback.on_step()

            if step < self.learning_starts:
                action = env.action_space.sample()
            else:
                action = self.policy.get_action(obs)

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
        s_t = transition.obs

        a_t, log_prob = self.policy.actor(s_t)
        alpha = self.alpha

        critic_1 = self.policy.critic_1(s_t, a_t)
        critic_2 = self.policy.critic_2(s_t, a_t)
        critic = th.min(critic_1, critic_2)

        loss = (alpha * log_prob - critic).mean()
        return loss
