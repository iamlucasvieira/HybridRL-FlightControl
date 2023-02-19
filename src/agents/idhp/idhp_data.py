"""Module that defines objects that store IDHP data."""
from dataclasses import dataclass
import numpy as np


@dataclass
class IDHPLearningData:
    """Class that stores the data used for learning."""
    action: np.ndarray
    obs: np.ndarray
    reward: float
    done: bool
