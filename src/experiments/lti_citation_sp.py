"""Experiment to learn using wandb while using the citation model in sp mode."""

from experiments.core import Experiment


def main(algorithm_name: str = "SAC",
         env_name: str = "citation",
         configuration: str = "sp",
         task_name="aoa_sin",
         seed: int = 1,
         dt: float = 0.1,
         episode_steps: int = 100,
         learning_steps: int = 10_000,
         verbose: int = 2,
         name=None,
         tags=None,
         TO_PLOT=False,
         offline=True):
    """Main function to run a basic aoa experiementthe experiment.

    args:
        algorithm_name: Name of the algorithm used.
        env_name: Environment to use.

    """

    exp = Experiment(algorithm_name=algorithm_name,
                     env_name=env_name,
                     configuration=configuration,
                     task_name=task_name,
                     seed=seed,
                     dt=dt,
                     episode_steps=episode_steps,
                     learning_steps=learning_steps,
                     verbose=verbose,
                     offline=offline
                     )

    # Learn the model
    exp.learn(name=name, tags=tags)

    # Run the model
    exp.run()

    # Plot results
    if TO_PLOT:
        exp.plot()


if __name__ == "__main__":
    main(TO_PLOT=True)
