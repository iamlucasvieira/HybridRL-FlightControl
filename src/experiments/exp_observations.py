"""Experiment to analyse the difference in performance for different observation types."""

from experiments.core import Experiment
import numpy as np

seed = None
project_name = "citation-SAC-observations"
learning_steps = 5_000

# Exp 1: Observation = all states + reference + tracking error
for _ in range(5):
    exp_1 = Experiment(
        project_name=project_name,
        seed=seed,
        learning_steps=learning_steps,
    )

    exp_1.learn(wandb_config={"obs": "all-states"})

    exp_1.finish_wandb()

    # Exp 2: Observation = tracking error
    exp_2 = Experiment(
        project_name=project_name,
        seed=seed,
        learning_steps=learning_steps,

    )

    env_2 = exp_2.env
    # exp_2.env._get_obs_shape = lambda self: (1,)
    env_2._get_obs = lambda: np.array([0 if not env_2.sq_error else env_2.sq_error[-1]]).astype(np.float32)

    exp_2.learn(wandb_config={"obs": "tracking-error"})

    exp_2.finish_wandb()
