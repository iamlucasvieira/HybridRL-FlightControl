"""Experiment to analyse the difference in performance for different rewards."""

from experiments.core import Experiment
import numpy as np
from helpers.config import ConfigLinearAircraft

project_name = "citation-SAC-rewards"
experiment_config = ConfigLinearAircraft(
    algorithm="SAC",
    configuration="sp",
    task="q_sin",
    seed=None,
    learning_steps=10_000,
)

for _ in range(1):

    for reward_scale in np.linspace(0.1, 1, 10):
        exp = Experiment(config=experiment_config, project_name=project_name)
        exp.env.reward_scale = reward_scale
        exp.learn(wandb_config={"reward_scale": reward_scale})
        exp.finish_wandb()
