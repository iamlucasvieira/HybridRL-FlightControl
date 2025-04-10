"""Defines the RL tasks functions."""

from math import ceil

import numpy as np


def get_reference_signal(reference_name: str):
    """Returns the task function."""

    if reference_name not in reference_dict:
        raise ValueError(f"Reference signal '{reference_name}' not found.")
    else:
        return reference_dict[reference_name]


class ReferenceSignals:
    """Class that defines the reference signals for the tracking task"""

    @staticmethod
    def step(env):
        """Task to track a step reference signal."""
        reference = 0.1

        return np.array([reference])

    @staticmethod
    def sin(env):
        """Task to track a sinusoidal reference signal."""
        period = 4 * np.pi
        length = env.episode_length
        amplitude = 0.1

        reference = amplitude * np.sin(period * env.current_time / length)

        return np.array([reference])

    @staticmethod
    def rectangle(env):
        """Task to track a rectangle reference signal."""
        reference = 0.1
        length = env.episode_length
        rectangle_length = length / 4

        if env.current_time < rectangle_length or env.current_time > (
            length - rectangle_length
        ):
            reference = 0

        return np.array([reference])

    @staticmethod
    def square_wave(env):
        """Task to track a square wave signal."""
        reference = -0.1
        length = env.episode_length
        square_length = length / 4

        if env.current_time <= square_length or (
            (env.current_time > 2 * square_length)
            and env.current_time <= 3 * square_length
        ):
            reference = 0.1

        return np.array([reference])

    @staticmethod
    def constant_sin(env):
        """Task to track a constant sinusoidal that repeats every 3 seconds."""
        period = 2 * np.pi / 3
        amplitude = 0.1

        reference = amplitude * np.sin(env.current_time * period)

        return np.array([reference])

    @staticmethod
    def constant_square(env):
        """Task to track a constant square wave that repeats every 3 seconds."""
        reference = -0.1
        start = 0
        middle = 2
        end = 4

        periods = ceil(env.current_time / (end - start))

        if env.current_time <= middle + (end - start) * (periods - 1):
            reference = 0.1
        else:
            reference = -0.1

        return np.array([reference])

    @staticmethod
    def sin_to_constant(env):
        """Task to track a sinusoidal reference signal that becomes constant at a certain time."""
        period = 2 * np.pi / 3
        length = env.episode_length
        amplitude = 0.1

        if period * env.current_time / (2 * np.pi) < 3:
            reference = amplitude * np.sin(period * env.current_time)
        else:
            reference = amplitude

        return np.array([reference])

    @staticmethod
    def sum_sin(env):
        """Task to track a sinusoidal reference signal."""
        period_1 = 2 * np.pi / 4
        period_2 = 2 * np.pi / 5

        amplitude = 0.1

        reference = amplitude * np.sin(
            env.current_time * period_1
        ) + amplitude * np.sin(env.current_time * period_2)

        return np.array([reference])


reference_dict = {
    "step": ReferenceSignals.step,
    "sin": ReferenceSignals.sin,
    "rectangle": ReferenceSignals.rectangle,
    "square_wave": ReferenceSignals.square_wave,
    "constant_sin": ReferenceSignals.constant_sin,
    "constant_square": ReferenceSignals.constant_square,
    "sin_to_constant": ReferenceSignals.sin_to_constant,
    "sum_sin": ReferenceSignals.sum_sin,
}

AVAILABLE_REFERENCES = list(reference_dict.keys())
