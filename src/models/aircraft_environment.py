"""Creates a gym environment for the aircraft model"""
import gym
from gym import spaces
import numpy as np

from models.aircraft_model import Aircraft
from models.tasks import get_task

def sig_const(dt):
    return 0.1

# def sig_const(length, step):
#     np.sin(np.arange(0, length, step)
#     return np.sin(time)

class AircraftEnv(gym.Env):
    """Aircraft Environment that follows gym interface"""
    metadata = {"render.modes": ["human"]}

    def __init__(self, config, *args, **kwargs):
        super(AircraftEnv, self).__init__()
        self.dt = config.dt
        self.episode_steps = config.episode_steps
        self.episode_length = self.episode_steps * self.dt
        self.task = get_task(config.task)

        self.aircraft = Aircraft(*args, dt=self.dt, **kwargs)

        self.current_time = 0
        self.reference = []
        self.track = []
        self.actions = []

        self.action_space = spaces.Box(low=-0.5, high=0.5,
                                       shape=(self.aircraft.ss.ninputs,), dtype=np.float32)

        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(self.aircraft.ss.nstates + 2,), dtype=np.float32)

    def step(self, action):

        self.current_time += self.dt
        self.actions.append(action)

        states = self.aircraft.response(action)

        states, reward, done, info = self.task(states.flatten(), action, self)

        reference = self.reference[-1]
        track = self.track[-1]
        tracking_error = (reference - track) ** 2

        # Build observation with reference value  and tracking error
        observation = np.append(states, [reference, tracking_error]).astype(np.float32)
        return observation, reward, done, info

    def reset(self):

        self.current_time = 0
        self.reference = []
        self.track = []
        self.actions = []

        # Reset the state of the environment to an initial state
        self.aircraft.build_state_space()

        #  Get initial state
        states = self.aircraft.current_state.flatten()

        observation = np.append(states, [0, 0]).astype(np.float32)

        return observation  # reward, done, info can't be included

    def render(self, mode="human"):
        # print("i")
        pass

    def close(self):
        pass
