"""Defines the RL tasks functions."""

import numpy as np


def get_task(task_name):
    """Returns the task function."""
    if task_name not in task_dict:
        raise ValueError(f"Task {task_name} not found.")
    else:
        return task_dict[task_name]


class LinearTasks:
    """Class that defines task functions for the LTI aircraft."""

    @staticmethod
    def step_aoa(env):
        """Step angle of attack reference."""
        return LinearTasks.track_step(env, state="alpha")

    @staticmethod
    def step_q(env):
        """Step pitch rate reference."""
        return LinearTasks.track_step(env, state="q")

    @staticmethod
    def sin_aoa(env):
        """Sinusoidal angle of attack reference."""
        return LinearTasks.track_sin(env, state="alpha")

    @staticmethod
    def sin_q(env):
        """Sinusoidal pitch rate reference."""
        return LinearTasks.track_sin(env, state="q")

    @staticmethod
    def rect_aoa(env):
        """Rectangular angle of attack reference."""
        return LinearTasks.track_rectangle(env, state="alpha")

    @staticmethod
    def rect_q(env):
        """Rectangular pitch rate reference."""
        return LinearTasks.track_rectangle(env, state="q")

    @staticmethod
    def square_aoa(env):
        """Square wave angle of attack reference."""
        return LinearTasks.track_square_wave(env, state="alpha")

    @staticmethod
    def square_q(env):
        """Square wave pitch rate reference."""
        return LinearTasks.track_square_wave(env, state="q")

    @staticmethod
    def track_step(env, state="q"):
        """Task to track a step reference signal."""
        reference = 0.1

        state_idx = env.aircraft.ss.x_names.index(state)
        state_value = env.aircraft.current_state.flatten()[state_idx]

        return state_value, reference

    @staticmethod
    def track_sin(env, state="q"):
        """Task to track a sinusoidal reference signal."""
        period = 4 * np.pi
        length = env.episode_length
        amplitude = 0.1

        reference = amplitude * np.sin(period * env.current_time / length)

        state_idx = env.aircraft.ss.x_names.index(state)
        state_value = env.aircraft.current_state.flatten()[state_idx]

        return state_value, reference

    @staticmethod
    def track_rectangle(env, state="q"):
        """Task to track a rectangle reference signal."""
        reference = 0.1
        length = env.episode_length
        rectangle_length = length / 4

        if env.current_time < rectangle_length or env.current_time > (length - rectangle_length):
            reference = 0

        state_idx = env.aircraft.ss.x_names.index(state)
        state_value = env.aircraft.current_state.flatten()[state_idx]

        return state_value, reference

    @staticmethod
    def track_square_wave(env, state="q"):
        """Task to track a square wave signal."""
        reference = -0.1
        length = env.episode_length
        square_length = length / 4

        if env.current_time <= square_length or (
                (env.current_time > 2 * square_length) and env.current_time <= 3 * square_length):
            reference = 0.1

        state_idx = env.aircraft.ss.x_names.index(state)
        state_value = env.aircraft.current_state.flatten()[state_idx]

        return state_value, reference


task_dict = {
    "step_aoa": LinearTasks.step_aoa,
    "sin_aoa": LinearTasks.sin_aoa,
    "rect_aoa": LinearTasks.rect_aoa,
    "square_aoa": LinearTasks.square_aoa,
    "step_q": LinearTasks.step_q,
    "sin_q": LinearTasks.sin_q,
    "rect_q": LinearTasks.rect_q,
    "square_q": LinearTasks.square_q,
}

AVAILABLE_TASKS = list(task_dict.keys())
