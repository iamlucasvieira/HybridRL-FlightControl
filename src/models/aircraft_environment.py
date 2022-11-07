"""Creates a gym environment for the aircraft model"""
import gym
from gym import spaces
import numpy as np

from models.aircraft_model import Aircraft

# def reward_signal():
#     pass

class AircraftEnv(gym.Env):
    """Aircraft Environment that follows gym interface"""
    metadata = {"render.modes": ["human"]}

    def __init__(self, *args, dt=0.001, T=100, **kwargs):
        super(AircraftEnv, self).__init__()

        self.aircraft = Aircraft(*args, dt=dt, **kwargs)
        self.total_time = T
        self.current_time = 0
        self.dt = dt

        self.action_space = spaces.Box(low=-1, high=1,
                                            shape=(self.aircraft.ss.ninputs,), dtype=np.float32)

        self.observation_space = spaces.Box(low=-25, high=25,
                                            shape=(self.aircraft.ss.nstates,), dtype=np.float64)

    def step(self, action):

        self.current_time += self.dt

        states = self.aircraft.response(action)

        done = False
        observation = states.flatten()
        info = {}


        aoa = observation[1]

        reward = -(0.05 - aoa)**2

        if abs(aoa) > 0.5:
            reward = -100
            done = True

        if self.current_time > self.total_time:
            done = True

        return observation, reward, done, info

    def reset(self):

        self.current_time = 0

        # Reset the state of the environment to an initial state
        self.aircraft.build_state_space()

        #  Get initial state
        observation = self.aircraft.current_state.flatten()

        return observation  # reward, done, info can't be included

    def render(self, mode="human"):
        # print("i")
        pass

    def close(self):
        pass


a = AircraftEnv()
a.step(-0.005)
