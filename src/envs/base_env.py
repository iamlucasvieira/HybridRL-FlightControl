from abc import ABC, abstractmethod
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.observations import get_observation
from envs.rewards import get_reward
from tasks import get_task


class BaseEnv(gym.Env, ABC):
    """Aircraft Environment that follows gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        dt: float = 0.1,
        episode_steps: int = 100,
        reward_scale: float = 1.0,
        task_type: str = "sin_q",
        reward_type: str = "sq_error",
        observation_type: str = "states + ref + error",
    ):
        """Initialize the environment."""
        super().__init__()

        # Set parameters
        self.dt = dt
        self.episode_steps = episode_steps
        self.episode_length = episode_steps * dt
        self.reward_scale = reward_scale

        # Set spaces
        self.action_space = self._action_space()

        self.task_type = task_type

        # Initialize data storage
        self.current_time = None
        self.reference = None
        self.track = None
        self.states = None
        self.actions = None
        self.sq_error = None
        self.error = None
        self.initialize()

        # Set reward, observation and reference functions
        self.get_reward = None
        self.get_obs = None
        self.set_reward_function(reward_type)
        self.set_observation_function(observation_type)

    @property
    def n_states(self):
        """The number of states of the environment."""
        raise NotImplementedError

    @property
    def n_inputs(self):
        """The number of inputs of the environment."""
        raise NotImplementedError

    @property
    def aircraft_states(self):
        """The states of the aircraft."""
        raise NotImplementedError

    @property
    def current_aircraft_state(self):
        """The current state of the aircraft."""
        raise NotImplementedError

    def _action_space(self) -> spaces.Box:
        """The action space of the environment."""
        raise NotImplementedError

    def state_transition(self, action):
        """The state transition function of the environment."""
        raise NotImplementedError

    def _check_constraints(self, reward, done, info):
        """Check if the constraints are met."""
        raise NotImplementedError

    def _initial_state(self) -> np.ndarray:
        """The initial state of the environment."""
        raise NotImplementedError

    def _reset(self):
        """Allows child classes to reset their parameters."""
        raise NotImplementedError

    def _observation_space(self) -> spaces.Box:
        """The observation space of the environment."""
        return spaces.Box(low=-1, high=1, shape=self._get_obs_shape(), dtype=np.float32)

    def step(self, action: np.ndarray) -> tuple:
        info = {}

        # Advance time
        self.current_time += self.dt

        x_t1 = self.state_transition(action)  # All states of the aircraft
        tracked_x_t1 = x_t1[self.task.mask]  # Values of the states being tracked
        x_t_r1 = self.task.reference()  # Reference value for the next state

        # Tracking error
        e = (tracked_x_t1 - self.reference[-1]) * self.task.scale
        e_2 = e**2

        # Store values
        self.actions.append(action)
        self.reference.append(x_t_r1)
        self.track.append(tracked_x_t1)
        self.states.append(x_t1)
        self.error.append(e)
        self.sq_error.append(e_2)

        terminated = False
        truncated = False
        reward = self.get_reward(self)

        reward, terminated, info = self._check_constraints(reward, terminated, info)

        if self.current_time + self.dt > self.episode_length:
            terminated = True
            info = {
                "message": f"Episode length exceeded: {self.current_time} = {self.episode_length}"
            }

        # Make sure reward is not an array
        if isinstance(reward, np.ndarray):
            reward = reward.item()

        observation = self.get_obs(self)

        return observation, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None):
        """Resets the environment."""
        super().reset(seed=seed)
        self.initialize()
        self._reset()
        observation = self.get_obs(self)
        return observation, {}

    def initialize(self):
        """Initializes the environment."""
        # Set reference signal
        self.set_task(self.task_type)

        self.current_time = 0
        self.actions = [np.zeros(self.action_space.shape[0])]
        self.error = [np.zeros(self.task.mask.sum())]
        self.sq_error = self.error.copy()

        #  Get initial state
        x_t_0 = self._initial_state()
        x_t_r_0 = self.task.reference()
        tracked_x_t_0 = x_t_0[self.task.mask]

        self.states = [x_t_0]
        self.reference = [x_t_r_0]
        self.track = [tracked_x_t_0]

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def _get_obs_shape(self):
        """Returns the shape of the observation."""
        return self.get_obs(self).shape

    def set_task(self, task_type: str) -> None:
        """Sets the reference signal function to be used."""
        self.task = get_task(task_type)(self)

    def set_reward_function(self, reward_type: str) -> None:
        self.get_reward = get_reward(reward_type)

    def set_observation_function(self, observation_type: str) -> None:
        self.get_obs = get_observation(observation_type)
        self.observation_space = self._observation_space()

    @property
    def nmae(self):
        """Normalized mean absolute error."""
        reference = np.array(self.reference)
        track = np.array(self.track)
        error = reference - track
        observation_ranges = reference.max(axis=0) - reference.min(axis=0)
        observation_ranges[observation_ranges <= 0] = np.deg2rad(5) - np.deg2rad(-5)

        return np.mean(np.mean(np.abs(error), axis=0) / observation_ranges)

    @property
    @abstractmethod
    def states_name(self):
        """The names of the states."""
        raise NotImplementedError
