# -*- coding: utf-8 -*-
import pathlib as pl
from typing import Optional

import typer
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt, Confirm
from rich.table import Table

from helpers.paths import Path
from hrl_fc.experiment_runner import Runner, Evaluator

app = typer.Typer()
console = Console()


def available_experiment_files():
    """Returns a list of available experiment files."""
    return [f.name for f in Path.exp.iterdir() if f.is_file() and f.suffix == ".yaml"]


@app.command()
def main(
    filename: Optional[str] = typer.Argument(None, help="Experiment file name"),
    filepath: Optional[pl.Path] = typer.Argument(Path.exp, help="Experiment file path"),
    offline: Optional[bool] = typer.Option(False, help="Run experiment offline"),
):
    """Runs an experiment from a config file."""
    if filename is None:
        table = Table(header_style="bold magenta")
        table.add_column("Id", justify="left", style="green", no_wrap=True)
        table.add_column("Experiment", justify="left", style="green")
        experiments = available_experiment_files()
        for idx, file in enumerate(experiments):
            table.add_row(str(idx + 1), file)

        panel = Panel(
            table,
            title="Available experiments to run",
            title_align="center",
            border_style="magenta",
            expand=False,
        )
        console.print(panel)

        if Confirm.ask("Do you want to run an experiment? :test_tube:"):
            idx = IntPrompt.ask(
                "Select an experiment :robot:",
                choices=[str(i) for i in range(1, len(experiments) + 1)],
            )
            filename = experiments[idx - 1]
        else:
            raise typer.Exit()
    filename = filename + ".yaml" if not filename.endswith(".yaml") else filename
    file = filepath / filename
    if not file.exists():
        print(f"File {file} does not exist :sweat:")
        raise typer.Abort()

    Runner(file_name=filename, file_path=filepath).run()


@app.command()
def eval(
    model_directory: Optional[str] = typer.Argument(
        None, help="Directory of the model to evaluate"
    ),
    models_directory: Optional[pl.Path] = typer.Argument(
        Path.models, help="Models path"
    ),
):
    """Evaluates an experiment from a zip file."""
    print("Evaluating experiment...")
    try:
        evaluator = Evaluator(
            model_directory=model_directory, models_directory=models_directory
        )
        evaluator.evaluate()
        print("Evaluation finished :tada:")
    except FileNotFoundError:
        print(f"Path {model_directory / model_directory} does not exist :sweat:")
    except ValueError:
        print(f"Not able to load model from {model_directory} :sweat:")


if __name__ == "__main__":
    typer.run(main)
