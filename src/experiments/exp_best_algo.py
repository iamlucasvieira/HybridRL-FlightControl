"""Experiment to determine the best algorithm for the linear aircraft environment."""

from experiments.core import Experiment
from agents.seres_dsac import DSAC
from helpers.config import ConfigLinearAircraft
from models.aircraft_environment import AircraftEnv

def main(task="q_sin", project_name="best_algo", episode_steps=100, learning_steps=10_000, run=1, iterations=1):
    """Run the experiment."""
    for _ in range(iterations):
        # np.random.seed(0)
        exp_sac = Experiment(
            project_name=project_name,
            algorithm_name="SAC",
            task_name=task,
            episode_steps=episode_steps,
            learning_steps=learning_steps,
            run=run,
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
        )

        exp_td3.learn()
        exp_td3.finish_wandb()

        config = ConfigLinearAircraft(
            task=task,
            algorithm="DSAC",
            episode_steps=episode_steps,
            learning_steps=learning_steps,

        )

        env = AircraftEnv(config)

        dsac = DSAC(env, config, project_name=project_name)
        dsac.train()

if __name__ == '__main__':
    main(iterations=10)
