"""Module that defines the base agent that is stable-baselines3 like."""
from abc import ABC, abstractmethod
import gymnasium as gym
from agents.base_callback import ListCallback, BaseCallback
from agents.base_policy import BasePolicy
from helpers.torch_helpers import get_device
from typing import Optional, List, Tuple, Any, Union, SupportsFloat
import random
import torch as th
import numpy as np
import time
from agents.base_logger import Logger
import pathlib as pl
from hrl_fc.console import console
import pickle
from agents.buffer import ReplayBuffer, Transition
from envs.base_env import BaseEnv

class BaseAgent(ABC):
    """Base agent class."""

    def __init__(self, policy: BasePolicy, env: Union[BaseEnv, gym.Env],
                 device: Optional[str] = None,
                 verbose: int = 0,
                 seed: Optional[int] = None,
                 log_dir: Optional[str] = None,
                 save_dir: Optional[str] = None,
                 policy_kwargs: Optional[dict] = None,
                 _init_setup_model: bool = True, ) -> None:

        """Initialize the base agent.

        Args:
            env: Environment.
        """
        self.env = env
        self.device = get_device() if device is None else device
        self.verbose = verbose
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.log_dir = log_dir
        self.save_dir = save_dir
        self.seed = seed
        self.logger = None

        # Setup data
        self.policy = None
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._n_updates = 0
        self.run_name = None
        self.episode_buffer = None
        self.policy = policy(self.observation_space, self.action_space, **self.policy_kwargs)

        # Setup learn variables
        self.total_steps = None
        self.num_steps = None
        self.start_time = None
        self._episode_num = None

        # Print initialized agent
        self.print(f"Using device: {self.device}")

        if _init_setup_model:
            self._setup_model()

    def print(self, message: str, identifier=True, **kwargs):
        """Print only if verbosity is enabled."""
        if self.verbose > 0:
            console.print(f"{'Agent: ' if identifier else ''}{message}", style="blue", **kwargs)

    def _setup_model(self):
        """Set up the model."""
        self.print("Setting up model:", end=" ")
        self.set_random_seed()
        self.setup_model()
        self.print("Done :heavy_check_mark:", identifier=False)

    def get_rollout(self, action: np.ndarray, obs: np.ndarray, callback: ListCallback) \
            -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """Get the rollout.

        Args:
            action: Action to take.
            obs: Observation.
            callback: Callbacks.
        """
        callback.on_rollout_start()

        # Get the rollout
        obs_tp1, reward, terminated, truncated, info = self.env.step(action)

        callback.on_rollout_end()

        # Store rollout data
        rollout = Transition(obs=obs,
                             action=action,
                             reward=reward,
                             obs_=obs_tp1,
                             done=terminated or truncated, )

        self.episode_buffer.push(rollout)
        self.logger.record("rollout/episode_reward", reward)
        return obs, reward, terminated, truncated, info

    def _setup_learn(self,
                     total_steps: int,
                     run_name: str,
                     callback: Optional[List[BaseCallback]] = None,
                     buffer_factor: int = .2
                     ) -> ListCallback:
        """Set up the learn method.

        Args:
            total_steps: Total number of steps to learn.
            run_name: Name of the run.
            callback: Callbacks.
            buffer_factor: Factor to multiply the total steps by to get the size of the buffer.
        """
        # Set starting variables
        self.start_time = time.time_ns()
        self.num_steps = 0
        self.total_steps = total_steps
        self._episode_num = 0
        self.run_name = run_name

        # Set logger
        self.logger = Logger(self.log_dir, run_name, self.verbose)

        # Create buffer for episode data
        self.episode_buffer = ReplayBuffer(int(self.total_steps * buffer_factor))

        # Initialize callbacks
        callback = self._init_callback(callback)
        return callback

    @abstractmethod
    def _learn(self, total_steps: int, callback: ListCallback, log_interval: int) -> None:
        """Learn method."""
        pass

    def learn(self, total_steps: int, run_name: str = "run", callback: Optional[List[BaseCallback]] = None,
              log_interval: int = 4) -> None:
        """Learn method.

        Args:
            total_steps: Total number of steps to learn.
            run_name: Name of the run.
            callback: Callbacks.
            log_interval: Log interval.
        """
        # Set up learn
        callback = self._setup_learn(total_steps, run_name, callback=callback)
        callback.on_training_start(locals(), globals())

        # Run user defined learn
        self._learn(total_steps, callback, log_interval)

        callback.on_training_end()

    def dump_logs(self):
        """Dump logs."""
        # Log rollout data
        rollout = self.episode_buffer.sample_buffer()
        self.logger.record("rollout/reward_mean", np.mean(rollout.reward))
        self.logger.record("rollout/reward_std", np.std(rollout.reward))
        self.logger.record("rollout/n_episodes", self._episode_num)
        self.logger.record("rollout/total_steps", self.num_steps)
        self.logger.record("rollout/total_time", (time.time_ns() - self.start_time) / 1e9)
        self.logger.dump(step=self.num_steps)

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

    def save(self, path: Optional[pl.Path] = None):
        """Save the model."""
        if path is None:
            path = self.save_dir
        if self.run_name is None:
            raise ValueError("Run name not set. Agent needs to 'learn' before saving.")
        path = path / self.run_name

        policy_path = path / 'policy.pt'
        env_path = path / 'env.pkl'

        # Make directory for files
        path.mkdir(parents=True, exist_ok=True)

        th.save(self.policy.state_dict(), policy_path)

        # Save environment as pickle
        try:
            with open(env_path, 'wb') as f:
                pickle.dump(self.env, f)
        except Exception as e:
            self.print(f"Not able to pickle the environment '{e}'")
            # Delete file if created
            if env_path.is_file():
                env_path.unlink()

    def load(self, path: str):
        """Load the model.

        Args:
            path: Path to the model.
        """
        path = path
        policy_path = path / 'policy.pt'
        env_path = path / 'env.pkl'

        # Load the policy
        if policy_path.is_file():
            self.policy.load_state_dict(th.load(policy_path))
        else:
            raise ValueError("Policy file not found.")

        # Load the environment
        if env_path.is_file():
            with open(env_path, 'rb') as f:
                self.env = pickle.load(f)
        else:
            self.print("Environment file not found.")

    def predict(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Predict the action.

        Args:
            observation: Observation.
            deterministic: Whether to use deterministic action.
        """
        return self.policy.predict(observation, deterministic=deterministic)