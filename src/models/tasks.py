"""Defines the RL tasks functions."""
from dataclasses import dataclass

import numpy as np


def get_task(task_name):
    """Returns the task function."""
    linear_tasks = LinearTasks()

    task_dict = {
        "aoa": linear_tasks.track_aoa,
        "aoa_sin": linear_tasks.track_aoa_sin,
        "q": linear_tasks.track_q,
        "q_sin": linear_tasks.track_q_sin,
    }

    if task_name not in task_dict:
        raise ValueError(f"Task {task_name} not found.")
    else:
        return task_dict[task_name]


class LinearTasks:
    """Class that defines task functions for the LTI aircraft."""

    @staticmethod
    def track_aoa(env):
        return LinearTasks.track(env, state="alpha")

    @staticmethod
    def track_q(env):
        return LinearTasks.track(env, state="q")

    @staticmethod
    def track_aoa_sin(env):
        return LinearTasks.track_sin(env, state="alpha")

    @staticmethod
    def track_q_sin(env):
        return LinearTasks.track_sin(env, state="q")

    @staticmethod
    def track(env, state="alpha"):
        """Task to track angle of attack."""
        reference = 0.1

        state_idx = env.aircraft.ss.x_names.index(state)
        state_value = env.aircraft.current_state.flatten()[state_idx]

        env.reference.append(reference)
        env.track.append(state_value)
        env.sq_error.append((reference - state_value) ** 2)

        return state_value, reference

    @staticmethod
    def track_sin(env, state="alpha"):
        """Task to track angle of attack."""
        period = 4 * np.pi
        length = env.episode_length
        amplitude = 0.1

        reference = amplitude * np.sin(period * env.current_time / length)

        state_idx = env.aircraft.ss.x_names.index(state)
        state_value = env.aircraft.current_state.flatten()[state_idx]

        env.reference.append(reference)
        env.track.append(state_value)
        env.sq_error.append((reference - state_value) ** 2)

        return state_value, reference
