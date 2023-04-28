"""Module that runs an experiment from a configuration file."""
import os

import wandb
from rich.progress import track

from helpers.misc import verbose_print
from helpers.paths import Path
from helpers.wandb_helpers import evaluate
from hrl_fc.experiment_builder import ExperimentBuilder


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
                save_code=False,  # optional
                monitor_gym=True,
            )

            # Learn
            sweep.learn(name=wandb_run.name)

            config_path = None

            if self.config.save_model:
                config_path = self.experiment.file_path / self.experiment.filename
                sweep.save_model(config_path=config_path)

            # Evaluate
            if self.config.evaluate:
                sweep.evaluate(config_path)

            wandb_run.finish()

    def evaluate(self, sweep):
        """Evaluate a sweep."""
        self.print("Evaluating...")

        sweep_config = sweep.config
        wandb_run = wandb.init(
            project=sweep_config.name,
            config=sweep_config.dict(),
            sync_tensorboard=False,  # auto-upload sb3's tensorboard metrics
            save_code=False,  # optional
            monitor_gym=True,
        )
        sweep.evaluate()
        wandb_run.finish()

    def print(self, message: str):
        """Prints only if verbose is greater than 0."""
        verbose_print(message, self.config.verbose)

    @classmethod
    def from_file(cls, *args, **kwargs):
        """Load a model file."""
        # Initialize class
        runner = cls(*args, **kwargs)

        for sweep in runner.experiment.sweeps:
            sweep.load_model(runner.file_path)

        return runner


class Evaluator:
    """Class responsible for replaying evaluation of stored models."""

    def __init__(
        self,
        model_directory: str,
        models_directory: str = None,
        zip_name: str = None,
        verbose=0,
    ):
        """Initialize the evaluator."""
        os.environ["WANDB_DIR"] = Path.logs.as_posix()
        self.verbose = verbose

        self.print("Initializing evaluator...")
        self.agent, self.data = load_agent(
            model_directory=model_directory,
            models_directory=models_directory,
            zip_name=zip_name,
            with_data=True,
        )

    def evaluate(self):
        """Evaluate a sweep."""
        wandb_run = wandb.init(
            project="replay",
            config=self.data,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            save_code=False,  # optional
        )

        self.print("Evaluating...")
        evaluate(self.agent, self.agent.env)

        wandb_run.finish()

    def print(self, message: str):
        """Prints only if verbose is greater than 0."""
        verbose_print(message, self.verbose)


def main():
    Runner("exp_sac_citation").run()


if __name__ == "__main__":
    main()
