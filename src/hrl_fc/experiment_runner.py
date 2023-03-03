"""Module that runs an experiment from a configuration file."""
from hrl_fc.experiment_builder import ExperimentBuilder, Sweep
from rich.progress import track
import os
from helpers.paths import Path
import wandb


class Runner:
    """Class that runs an experiment."""

    def __init__(self, file_name, file_path=None):
        self.file_name = file_name
        self.file_path = file_path
        self.experiment = ExperimentBuilder(file_name, file_path=file_path)
        self.config = self.experiment.config

        # Set wandb
        self.wandb_run = None
        os.environ["WANDB_DIR"] = Path.logs.as_posix()
        if not self.config.wandb:
            os.environ["WANDB_MODE"] = "offline"

    def run(self):
        """Run an experiment."""
        self.print("Running experiment...")

        for sweep in track(self.experiment.sweeps, description=":brain: Learning..."):

            sweep_config = sweep.config

            wandb_run = wandb.init(
                project=sweep_config.name,
                config=sweep_config.dict(),
                sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                save_code=True,  # optional
            )

            # Learn
            sweep.learn(name=wandb_run.name)

            # Evaluate
            if self.config.evaluate:
                self.evaluate(sweep)

            wandb_run.finish()

    def learn(self, sweep: Sweep):
        """Start learning a sweep."""

    def evaluate(self):
        pass

    def print(self, message: str):
        """Prints only if verbose is greater than 0."""
        if self.config.verbose > 0:
            print(message)


def main():
    Runner('exp_idhp_hyperparameters').run()


if __name__ == "__main__":
    main()
