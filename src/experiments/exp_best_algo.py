"""Experiment to determine the best algorithm for the linear aircraft environment."""

from experiments.core import Experiment


def main(task="q_sin", configuration="sp", project_name="best_algo", episode_steps=100, learning_steps=10_000, run=1,
         iterations=1):
    """Run the experiment."""
    for _ in range(iterations):

        for algo in ["SAC", "TD3", "DSAC"]:
            exp = Experiment(
                project_name=project_name,
                algorithm_name=algo,
                task_name=task,
                configuration=configuration,
                episode_steps=episode_steps,
                learning_steps=learning_steps,
                run=run,
            )

            exp.learn()

            exp.finish_wandb()


if __name__ == '__main__':
    main(iterations=10)
