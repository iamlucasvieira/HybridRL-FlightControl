"""Experiment to analyse the difference in performance for different rewards."""

from experiments.core import Experiment
import numpy as np
from helpers.config import ConfigLinearAircraft

project_name = "best_reward"
experiment_config = ConfigLinearAircraft(
    algorithm="TD3",
    configuration="sp",
    task="q_sin",
    learning_steps=10_000,
    run=1
)

for _ in range(6):
    for reward_type in ["sq_error", "sq_error_da", "sq_error_da_a"]:
        experiment_config.reward_type = reward_type

        exp = Experiment(config=experiment_config,
                         project_name=project_name)

        exp.learn()
        exp.finish_wandb()
