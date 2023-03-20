"""Module with helper functions for the SB3 agents."""
import importlib
import pathlib as pl
from typing import Tuple
from typing import Union

from helpers.paths import Path


# from stable_baselines3.common.base_class import BaseAlgorithm
# from stable_baselines3.common.save_util import load_from_zip_file



def load_agent(
    model_directory: str,
    models_directory: str = None,
    zip_name: str = None,
    with_data=False,
):
    """Loads an agent."""
    file_path = Path.models if models_directory is None else pl.Path(models_directory)
    zip_name = "model.zip" if zip_name is None else zip_name
    file = file_path / model_directory / zip_name

    if not file.is_file():
        raise FileNotFoundError(f"File {file} does not exist.")

    data, _, _ = load_from_zip_file(file)
    policy_class = data["policy_class"]

    # Check if the policy used is implemented in this repo
    if not isinstance(policy_class, str):
        policy_class = policy_class.__module__
    if policy_class.startswith("agents."):
        algorithm_name = policy_class.split(".")[1]
    else:
        raise ValueError("Policy not implemented for re-loading")

    switch = {
        "sac": "SAC",
        "idhp_sac": "IDHPSAC",
    }

    if algorithm_name not in switch:
        raise ValueError("Algorithm not implemented for re-loading")

    algorithm = switch[algorithm_name]

    # Import agent
    agent = getattr(importlib.import_module("agents"), algorithm)

    # Load model
    agent = agent.load(file)

    if with_data:
        return agent, data
    return agent


if __name__ == "__main__":
    agent = load_agent("IDHP-SAC-LTI/robust-sky-24")
