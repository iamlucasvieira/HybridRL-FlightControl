"""Module that define configuration of algorithms."""

from dataclasses import dataclass, asdict


@dataclass
class ConfigLinearAircraft:
    """Class that defines the configuration of the linear aircraft."""

    # General

    policy_type: str = "MlpPolicy"
    env_name: str = "citation"
    filename: str = "citation.yaml"
    configuration: str = "symmetric"
    algorithm: str = ""
    seed: int = 1  # Random seed
    dt: float = 0.1  # Time step
    episode_steps: int = 100  # Number of steps
    learning_steps: int = 1000  # Number of total learning steps
    task: str = "aoa"

    @property
    def asdict(self):
        """Return class as dict"""
        return asdict(self)
