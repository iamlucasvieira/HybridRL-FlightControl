"""Creates a gym environment for the aircraft model"""
import gym
from gym import spaces
import numpy as np

from models.aircraft_model import Aircraft
from models.tasks import get_task


class AircraftEnv(gym.Env):
    """Aircraft Environment that follows gym interface"""
    metadata = {"render.modes": ["human"]}

    def __init__(self, config, *args, **kwargs):
        super(AircraftEnv, self).__init__()
        self.dt = config.dt
        self.episode_steps = config.episode_steps
        self.episode_length = self.episode_steps * self.dt
        self.task = get_task(config.task)
        self.configuration = config.configuration

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

        self.action_space = spaces.Box(low=-0.3, high=0.3,
                                       shape=(self.aircraft.ss.ninputs,), dtype=np.float32)

        obs_shape = self._get_obs_shape()
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=obs_shape, dtype=np.float32)

    def step(self, action):
        self.current_time += self.dt
        self.actions.append(action)

        self.current_states = self.aircraft.response(action).flatten()

        _, reward, done, info = self.task(self.current_states, action, self)

        observation = self._get_obs()

        return observation, reward, done, info

    def reset(self):
        self.current_time = 0
        self.reference = []
        self.track = []
        self.actions = []
        self.sq_error = []

        # Reset the state of the environment to an initial state
        self.aircraft.build_state_space()

        #  Get initial state
        states = self.aircraft.current_state.flatten()
        self.current_states = self.aircraft.current_state.flatten()

        observation = self._get_obs()

        return observation  # reward, done, info can't be included

    def render(self, mode="human"):
        # print("i")
        pass

    def close(self):
        pass

    def _get_obs(self):
        """Returns the current observation."""
        states = self.aircraft.current_state
        reference = 0 if not self.reference else self.reference[-1]
        tracking_error = 0 if not self.sq_error else self.sq_error[-1]
        return np.append(states, [reference, tracking_error]).astype(np.float32)

    def _get_obs_shape(self):
        """Returns the shape of the observation."""
        return (self.aircraft.ss.nstates + 2,)


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
