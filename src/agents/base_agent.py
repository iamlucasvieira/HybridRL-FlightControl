"""Module that defines the base agent that is stable-baselines3 like."""
from abc import ABC, abstractmethod
import gymnasium as gym
# from stable_baselines3.common.base_class import BaseAlgorithm
from agents.base_callback import ListCallback, BaseCallback
from helpers.torch_helpers import get_device
from typing import Optional, List
import random
import torch as th
import numpy as np
import time
from agents.base_logger import Logger

class BaseAgent(ABC):
    """Base agent class."""

    policy_aliases = {}

    def __init__(self, policy: str, env: gym.Env,
                 device: Optional[str] = None,
                 verbose: int = 0,
                 seed: Optional[int] = None,
                 policy_kwargs: Optional[dict] = None,
                 tensorboard_log: Optional[str] = None,
                 _init_setup_model: bool = True) -> None:

        """Initialize the base agent.

        Args:
            env: Environment.
        """
        self.policy_class = self.get_policy(policy)
        self.env = env
        self.device = get_device() if device is None else device
        self.verbose = verbose
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.tensorboard_log = tensorboard_log
        self.seed = seed
        self.logger = None

        # Setup data
        self.policy = None
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._n_updates = 0

        # Setup learn variables
        self.total_steps = None
        self.num_steps = None
        self.start_time = None
        self._episode_num = None

        # Print initialized agent
        self.print(f"Using device: {self.device}")

        if _init_setup_model:
            self._setup_model()

    def get_policy(self, policy: str):
        """Get the policy object from the policy aliases."""
        if policy not in self.policy_aliases:
            raise ValueError(f'Policy {policy} not supported.')
        return self.policy_aliases[policy]

    def print(self, message: str):
        """Print only if verbosity is enabled."""
        if self.verbose > 0:
            print(message)

    def _setup_model(self):
        """Set up the model."""
        self.print("Setting up model...")
        self.set_random_seed()
        self.setup_model()
        pass

    def _setup_learn(self,
                     total_steps: int,
                     callback: Optional[List[BaseCallback]] = None,
                     tb_log_name: str = "run", ) -> ListCallback:
        """Set up the learn method."""
        self.start_time = time.time_ns()
        self.num_steps = 0
        self.total_steps = total_steps
        self._episode_num = 0
        self.logger = Logger(self.tensorboard_log, tb_log_name, self.verbose)
        callback = self._init_callback(callback)
        return callback

    @abstractmethod
    def _learn(self, total_steps: int, callback: List[BaseCallback], log_interval: int) -> None:
        """Learn method."""
        pass

    def learn(self, total_steps: int, callback: Optional[List[BaseCallback]] = None,
              log_interval: int = 4) -> None:
        """Learn method.

        Args:
            total_steps: Total number of steps to learn.
            callback: Callbacks.
            log_interval: Log interval.
        """

        callback = self._setup_learn(total_steps, callback=callback)
        callback.on_training_start(locals(), globals())
        self._learn(total_steps, callback, log_interval)

    def _init_callback(self, callback: List[BaseCallback]) -> ListCallback:
        """Initialize the callback."""
        callback = ListCallback(callback)
        callback.init_callback(self)
        return callback

    def set_random_seed(self) -> None:
        """Set the random seed."""
        seed = self.seed

        if seed is None:
            return

        # Seed python RNG
        random.seed(seed)

        # Seed numpy RNG
        np.random.seed(seed)

        # seed the RNG for all devices (both CPU and CUDA)
        th.manual_seed(seed)

        # seed action space
        self.action_space.seed(seed)

        # seed environment
        self.env.reset(seed=seed)

    @abstractmethod
    def setup_model(self):
        """Agent specific setup."""
        pass
