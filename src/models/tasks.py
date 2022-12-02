"""Defines the RL tasks functions."""
from dataclasses import dataclass

import numpy as np


def get_task(task_name):
    """Returns the task function."""
    linear_tasks = LinearTasks()

    task_dict = {
        "aoa": linear_tasks.track_aoa,
        "aoa_sin": linear_tasks.track_aoa_sin,
    }

    if task_name not in task_dict:
        raise ValueError(f"Task {task_name} not found.")
    else:
        return task_dict[task_name]


class LinearTasks:
    """Class that defines task functions for the LTI aircraft."""

    @staticmethod
    def track_aoa(observation, action, env):
        """Task to track angle of attack."""

        del action  # unused

        reference = 0.1

        aoa_idx = env.aircraft.ss.x_names.index("alpha")
        aoa = observation[aoa_idx]

        done = False
        reward = - (reference - aoa) ** 2
        info = {}

        if abs(aoa) > 0.5:
            reward *= 100
            done = True

        if env.current_time > env.episode_length:
            done = True

        env.reference.append(reference)
        env.track.append(aoa)
        env.sq_error.append((reference - aoa) ** 2)

        return observation, reward, done, info,

    @staticmethod
    def track_aoa_sin(observation, action, env):
        """Task to track angle of attack."""

        del action

        period = 4 * np.pi
        length = env.episode_length
        amplitude = 0.1

        reference = amplitude * np.sin(period * env.current_time / length)

        aoa_idx = env.aircraft.ss.x_names.index("alpha")
        aoa = observation[aoa_idx]

        done = False
        reward = - (reference - aoa) ** 2
        info = {}

        if abs(aoa) > 0.5:
            reward *= 100
            done = True

        if env.current_time > env.episode_length:
            done = True

        env.reference.append(reference)
        env.track.append(aoa)

        return observation, reward, done, info,
