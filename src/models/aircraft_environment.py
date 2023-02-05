"""Creates a gym environment for the aircraft model"""
import gym
from gym import spaces
import numpy as np

from models.aircraft_model import Aircraft
from models.tasks import get_task
from models.rewards import get_reward
from models.observations import get_observation

class AircraftEnv(gym.Env):
    """Aircraft Environment that follows gym interface"""
    metadata = {"render.modes": ["human"]}

    def __init__(self, config, *args, **kwargs):
        super(AircraftEnv, self).__init__()
        self.dt = config.dt
        self.episode_steps = config.episode_steps
        self.episode_length = self.episode_steps * self.dt

        self.task = get_task(config.task_type)
        self._get_reward = get_reward(config.reward_type)
        self._get_obs = get_observation(config.observation_type)

        self.configuration = config.configuration
        self.reward_scale = config.reward_scale

        self.aircraft = Aircraft(*args,
                                 filename=config.filename,
                                 dt=self.dt,
                                 configuration=self.configuration,
                                 **kwargs)
        self.current_states = None

        self.current_time = 0
        self.reference = []
        self.track = []
        self.actions = []
        self.sq_error = []
        self.error = []

        self.action_space = spaces.Box(low=-0.3, high=0.3,
                                       shape=(self.aircraft.ss.ninputs,), dtype=np.float32)

        obs_shape = self._get_obs_shape()
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=obs_shape, dtype=np.float32)

    def step(self, action):
        # Advance time
        self.current_time += self.dt

        # Get aircraft response and the task results
        self.current_states = self.aircraft.response(action).flatten()
        state_value, reference = self.task(self)

        # Store values
        self.actions.append(action)
        self.reference.append(reference)
        self.track.append(state_value)
        self.sq_error.append((reference - state_value) ** 2)
        self.error.append(reference - state_value)

        done = False

        reward = self._get_reward(self)

        if abs(state_value) > 0.5:
            reward *= 100
            done = True

        if self.current_time > self.episode_length:
            done = True

        observation = self._get_obs(self)
        info = {}

        return observation, reward, done, info

    def update_observation_space(self):
        """Updates the observation space."""
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=self._get_obs_shape(), dtype=np.float32)
    def reset(self):
        self.current_time = 0
        self.reference = []
        self.track = []
        self.actions = []
        self.sq_error = []
        self.error = []

        # Reset the state of the environment to an initial state
        self.aircraft.build_state_space()

        #  Get initial state
        states = self.aircraft.current_state.flatten()
        self.current_states = self.aircraft.current_state.flatten()

        observation = self._get_obs(self)

        return observation  # reward, done, info can't be included

    def render(self, mode="human"):
        # print("i")
        pass

    def close(self):
        pass

    def _get_obs_shape(self):
        """Returns the shape of the observation."""
        return self._get_obs(self).shape

class AircraftIncrementalEnv(AircraftEnv):
    """Incremental model of the Aircraft"""
    metadata = {"render.modes": ["human"]}

    def __init__(self, config, *args, **kwargs):
        super(AircraftIncrementalEnv, self).__init__(config, *args, **kwargs)

        self.action_space = spaces.Box(low=-0.5, high=0.5,
                                       shape=(self.aircraft.ss.ninputs,), dtype=np.float32)

        obs_shape = self._get_obs_shape()
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=obs_shape, dtype=np.float32)
