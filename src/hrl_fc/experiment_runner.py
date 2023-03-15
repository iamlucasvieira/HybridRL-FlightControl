"""Module that runs an experiment from a configuration file."""
import importlib
import os
import pathlib as pl

import wandb
from rich.progress import track
from stable_baselines3.common.save_util import load_from_zip_file

from helpers.misc import verbose_print
from helpers.paths import Path
from hrl_fc.experiment_builder import ExperimentBuilder, Sweep


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
            )

            # Learn
            sweep.learn(name=wandb_run.name)

            if self.config.save_model:
                sweep.save_model()

            # Evaluate
            if self.config.evaluate:
                sweep._evaluate()

            wandb_run.finish()

    def evaluate(self, sweep):
        """Evaluate a sweep."""
        self.print("Evaluating...")
        obs = sweep.env.reset()
        while True:
            action, _states = sweep.agent.predict(obs, deterministic=True)
            obs, reward, done, info = sweep.env.step(action)
            sweep.env.render()
            if done:
                obs = sweep.env.reset()

    def print(self, message: str):
        """Prints only if verbose is greater than 0."""
        verbose_print(message, self.config.verbose)


class Evaluator:
    """Class responsible for replaying evaluation of stored models."""

    def __init__(self, file_name: str, file_path: str = None, zip_name: str = None, verbose=0):
        """Initialize the evaluator."""
        self.verbose = verbose

        self.print("Initializing evaluator...")
        file_path = Path.models if file_path is None else pl.Path(file_path)
        zip_name = "model.zip" if zip_name is None else zip_name
        file = file_path / file_name / zip_name

        if not file.is_file():
            raise FileNotFoundError(f"File {file} does not exist.")

        data, _, _ = load_from_zip_file(file)
        policy_class = data["policy_class"]
        self.data = data
        # Check if the policy used is implemented in this repo
        if policy_class.__module__.startswith("agents."):
            algorithm_name = policy_class.__module__.split(".")[1]
        else:
            raise ValueError("Policy not implemented for re-loading")

        switch = {
            "sac": "SAC",
        }

        if algorithm_name not in switch:
            raise ValueError("Algorithm not implemented for re-loading")

        algorithm = switch[algorithm_name]

        # Import agent
        agent = getattr(importlib.import_module("agents"), algorithm)

        # Load model
        self.agent = agent.load(file)

    def evaluate(self):
        """Evaluate a sweep."""
        wandb_run = wandb.init(
            project="replay",
            config=self.data,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            save_code=False,  # optional
        )

        self.print("Evaluating...")
        Sweep.evaluate(self.agent, self.agent._env)

        wandb_run.finish()

    def print(self, message: str):
        """Prints only if verbose is greater than 0."""
        verbose_print(message, self.verbose)


def main():
    Runner('exp_sac_lti').run()


if __name__ == "__main__":
    main()
