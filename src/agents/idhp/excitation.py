"""Module that defines the IDHP excitation functions."""
from math import exp, log, pi, sin


def get_excitation_function(excitation_name: str):
    """Returns the excitation function."""
    if excitation_name not in AVAILABLE_EXCITATION_FUNCTIONS:
        raise ValueError(f"Excitation function '{excitation_name}' not found.")
    else:
        return AVAILABLE_EXCITATION_FUNCTIONS[excitation_name]


class Excitation:
    """Class that defines the excitation functions"""

    @staticmethod
    def step(env):
        """Excitation function that returns a step signal."""
        reference = 0
        if env.current_time < 1:
            reference = 0.15
        return reference

    @staticmethod
    def decay(env):
        """Exponentially decay sinusoidal excitation."""
        t = env.current_time
        half_life_time = 2
        amplitude = env.action_space.high[0] * 0.4
        omega = 2 * pi / 0.2
        tau = half_life_time / log(2)
        excitation = 0
        if env.current_time < 1:
            excitation = amplitude * exp(-t / tau) * sin(omega * t)
        return excitation


AVAILABLE_EXCITATION_FUNCTIONS = {
    "step": Excitation.step,
    "decay": Excitation.decay,
}
