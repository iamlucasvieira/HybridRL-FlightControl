import gym
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from typing import Union, Optional
import torch as th
from copy import deepcopy
from agents.sac.policy import SACPolicy
from agents.sac.buffer import ReplayBuffer
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
                 device: Optional[Union[th.device, str]] = None,
                 ):
        """Initialize the SAC algorithm."""
        if device is None:
            device = get_device()

        self.buffer_size = buffer_size
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

    def learn(self):
        pass

    def _setup_model(self) -> None:
        """Initialize the SAC policy and replay buffer"""
        self.set_random_seed(self.seed)
        self.policy = self.policy_class(self.observation_space,
                                        self.action_space,
                                        **self.policy_kwargs)
        self.critic_policy = deepcopy(self.policy)
        self.replay_buffer = ReplayBuffer(self.buffer_size)

    def get_critic_loss(self):
        """Get the critic loss."""
        pass

    def get_actor_loss(self):
        """Get the actor loss."""
        pass
