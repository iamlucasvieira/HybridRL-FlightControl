"""Creates a gym environment for the aircraft model"""
import gym
from gym import spaces
import numpy as np

from envs.lti_citation.lti_model import Aircraft
from envs.lti_citation.tasks import get_task
from envs.lti_citation.rewards import get_reward
from envs.lti_citation.observations import get_observation


class AircraftEnv(gym.Env):
    """Aircraft Environment that follows gym interface"""
    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 filename: str = "citation.yaml",
                 configuration: str = "sp",
                 dt: float = 0.1,
                 episode_steps: int = 100,
                 task_type: str = "sin_q",
                 reward_scale: float = 1.0,
                 reward_type: str = "sq_error",
                 observation_type: str = "states + ref + error",
                 ):
        super(AircraftEnv, self).__init__()

        self.dt = dt
        self.episode_steps = episode_steps
        self.episode_length = episode_steps * dt

        self.get_reference = get_task(task_type)
        self.reward_scale = reward_scale

        self.aircraft = Aircraft(filename=filename,
                                 dt=self.dt,
                                 configuration=configuration,
                                 task_type=task_type)
        self.current_states = None

        self.current_time = 0
        self.reference = []
        self.track = []
        self.actions = []
        self.sq_error = []
        self.error = []

        self.action_space = spaces.Box(low=-0.3, high=0.3,
                                       shape=(self.aircraft.ss.ninputs,), dtype=np.float32)

        self.x_t = None  # Aircraft state at current time
        self.x_t_r = None  # Aircraft state reference at current time
        self.tracked_x_t = None
        self.tracked_state_mask = self.aircraft.tracked_state_map.flatten()  # Mask to select the tracked state
        self.initialize()

        self._get_reward = None
        self._get_obs = None
        self.set_reward_function(reward_type)
        self.set_observation_function(observation_type)

    def step(self, action):

        info = {}

        # Advance time
        self.current_time += self.dt

        # Get aircraft next state after action and the reference value for the next state
        x_t1 = self.aircraft.response(action).flatten()
        tracked_x_t1 = x_t1[self.tracked_state_mask]
        x_t_r1 = self.get_reference(self)

        # Tracking error
        e = tracked_x_t1 - self.reference[-1]
        e_2 = e ** 2

        # Store values
        self.actions.append(action)
        self.reference.append(x_t_r1)
        self.track.append(tracked_x_t1)
        self.error.append(e)
        self.sq_error.append(e_2)

        done = False

        reward = self._get_reward(self)

        if abs(x_t_r1) > 0.5:
            reward *= 100
            done = True
            info = {"message": f"Reference too large: {x_t_r1} > 0.5"}

        if self.current_time + self.dt > self.episode_length:
            done = True
            info = {"message": f"Episode length exceeded: {self.current_time} = {self.episode_length}"}

        # Make sure reward is not an array
        if isinstance(reward, np.ndarray):
            reward = reward.item()

        observation = self._get_obs(self)

        return observation, reward, done, info

    def update_observation_space(self):
        """Updates the observation space."""
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=self._get_obs_shape(), dtype=np.float32)

    def reset(self):
        self.initialize()
        observation = self._get_obs(self)
        return observation  # reward, done, info can't be included

    def initialize(self):
        """Initializes the environment."""
        self.current_time = 0
        self.actions = [np.zeros(self.action_space.shape[0])]
        self.error = [np.zeros([np.sum(self.tracked_state_mask)])]
        self.sq_error = self.error.copy()

        # Reset the state of the environment to an initial state
        self.aircraft.build_state_space()

        #  Get initial state
        x_t = self.aircraft.current_state.flatten()
        x_t_r = self.get_reference(self)
        tracked_x_t = x_t[self.tracked_state_mask]

        self.reference = [x_t_r]
        self.track = [tracked_x_t]

    def render(self, mode="human"):
        # print("i")
        pass

    def close(self):
        pass

    def _get_obs_shape(self):
        """Returns the shape of the observation."""
        return self._get_obs(self).shape

    def set_reward_function(self, reward_type: str) -> None:
        self._get_reward = get_reward(reward_type)

    def set_observation_function(self, observation_type: str) -> None:
        self._get_obs = get_observation(observation_type)
        self.update_observation_space()
        obs_shape = self._get_obs_shape()
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=obs_shape, dtype=np.float32)


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
