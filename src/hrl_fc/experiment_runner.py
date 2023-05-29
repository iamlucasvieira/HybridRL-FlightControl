"""Module that runs an experiment from a configuration file."""
import os
import pathlib as pl

import wandb
from rich.progress import Progress, TextColumn, TimeElapsedColumn

from helpers.misc import verbose_print
from helpers.paths import Path
from hrl_fc.console import console
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

        for sweep in self.experiment.sweeps:
            sweep_config = sweep.config
            with Progress(
                *Progress.get_default_columns(),
                TimeElapsedColumn(),
                TextColumn("{task.completed} of {task.total}"),
                console=console
            ) as progress:
                sweep.learn_kwargs["callback"].append(
                    ("progress", {"progress": progress})
                )

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

    def evaluate(self, sweep, task=None):
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
        sweep.evaluate(task=task)
        wandb_run.finish()

    def print(self, message: str):
        """Prints only if verbose is greater than 0."""
        verbose_print(message, self.config.verbose)

    @classmethod
    def from_file(
        cls,
        experiment_name: str,
        run_name: str,
        policy_name: str,
        models_dir: str = None,
    ):
        """Load a model file."""

        models_dir = pl.Path(models_dir) if models_dir is not None else Path.models
        file_path = models_dir / experiment_name / run_name
        file_name = "config"

        # Initialize class
        runner = cls(file_name, file_path)

        for sweep in runner.experiment.sweeps:
            sweep.load_model(runner.file_path, run=policy_name)

        return runner


def main():
    Runner("exp_sac_citation").run()


if __name__ == "__main__":
    main()
