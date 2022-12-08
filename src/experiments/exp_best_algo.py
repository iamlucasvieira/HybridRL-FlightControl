"""Experiment to determine the best algorithm for the linear aircraft environment."""

from experiments.core import Experiment
import numpy as np


def main(task="aoa_sin", project_name="best_algo", episode_steps=100, learning_steps=1_000, run=1):
    """Run the experiment."""

    np.random.seed(0)
    exp_sac = Experiment(

        project_name=project_name,
        algorithm_name="SAC",
        task_name=task,
        episode_steps=episode_steps,
        learning_steps=learning_steps,
        run=run,
        seed=np.random.randint(1_000),
    )

    exp_sac.learn()

    exp_sac.finish_wandb()

    exp_td3 = Experiment(
        project_name=project_name,
        algorithm_name="TD3",
        task_name=task,
        episode_steps=episode_steps,
        learning_steps=learning_steps,
        run=run,
        seed=np.random.randint(1_000),
    )

    exp_td3.learn()
    exp_td3.finish_wandb()


if __name__ == '__main__':
    main()
