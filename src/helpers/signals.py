"""Module with helper signals that provide a value for a given time t."""
import numpy as np


def cos_step(t: float, t0: float, w: float):
    """Smooth cosine step function."""
    a = -(np.cos(1 / w * np.pi * (t - t0)) - 1) / 2
    if t0 >= 0:
        a = 0.0 if t < t0 else a
    a = 1.0 if t > t0 + w else a
    return a
