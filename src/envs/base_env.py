import gym
from gym import spaces

from abc import ABC
import numpy as np

from envs.reference_signals import get_reference_signal
from envs.observations import get_observation
from envs.rewards import get_reward


class BaseEnv(gym.Env, ABC):
    """Aircraft Environment that follows gym interface"""
    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 dt: float = 0.1,
                 episode_steps: int = 100,
                 reward_scale: float = 1.0,
                 tracked_state: str = "q",
                 reference_type: str = "sin",
                 reward_type: str = "sq_error",
                 observation_type: str = "states + ref + error",
                 ):
        """Initialize the environment."""
        super(BaseEnv, self).__init__()

        # Set parameters
        self.dt = dt
        self.episode_steps = episode_steps
        self.episode_length = episode_steps * dt
        self.reward_scale = reward_scale
        self.tracked_state = tracked_state

        # Set spaces
        self.action_space = self._action_space()

        # Set reference signal
        self.get_reference = None
        self.set_reference_signal(reference_type)

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
    def tracked_state_mask(self):
        """A mask that has the shape of the aircraft states and the value 1 in the tracked state."""
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

    def _observation_space(self) -> spaces.Box:
        """The observation space of the environment."""
        return spaces.Box(low=-1, high=1,
                          shape=self._get_obs_shape(), dtype=np.float32)

    def step(self, action: np.ndarray) -> tuple:

        info = {}

        # Advance time
        self.current_time += self.dt

        # Get aircraft next state after action and the reference value for the next state
        x_t1 = self.state_transition(action)
        tracked_x_t1 = x_t1[self.tracked_state_mask]
        x_t_r1 = self.get_reference(self)

        # Tracking error
        e = tracked_x_t1 - self.reference[-1]
        e_2 = e ** 2

        # Store values
        self.actions.append(action)
        self.reference.append(x_t_r1)
        self.track.append(tracked_x_t1)
        self.states.append(x_t1)
        self.error.append(e)
        self.sq_error.append(e_2)

        done = False

        reward = self.get_reward(self)

        reward, done, info = self._check_constraints(reward, done, info)

        if self.current_time + self.dt > self.episode_length:
            done = True
            info = {"message": f"Episode length exceeded: {self.current_time} = {self.episode_length}"}

        # Make sure reward is not an array
        if isinstance(reward, np.ndarray):
            reward = reward.item()

        observation = self.get_obs(self)

        return observation, reward, done, info

    def reset(self):
        self.initialize()
        observation = self.get_obs(self)
        return observation  # reward, done, info can't be included

    def initialize(self):
        """Initializes the environment."""
        self.current_time = 0
        self.actions = [np.zeros(self.action_space.shape[0])]
        self.error = [np.zeros([np.sum(self.tracked_state_mask)])]
        self.sq_error = self.error.copy()

        #  Get initial state
        x_t_0 = self._initial_state()
        x_t_r_0 = self.get_reference(self)
        tracked_x_t_0 = x_t_0[self.tracked_state_mask]

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

    def set_reference_signal(self, reference_type: str) -> None:
        """Sets the reference signal function to be used."""
        self.get_reference = get_reference_signal(reference_type)

    def set_reward_function(self, reward_type: str) -> None:
        self.get_reward = get_reward(reward_type)

    def set_observation_function(self, observation_type: str) -> None:
        self.get_obs = get_observation(observation_type)
        self.observation_space = self._observation_space()