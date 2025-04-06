import pathlib as pl
from typing import Optional

import typer
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt
from rich.table import Table

from helpers.paths import Path
from hybridrl_flightcontrol.experiment_runner import Runner


app = typer.Typer()
console = Console()


def available_experiment_files():
    """Returns a list of available experiment files."""
    return [f.name for f in Path.exp.iterdir() if f.is_file() and f.suffix == ".yaml"]


@app.command()
def run(
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
    experiment_name: str = typer.Argument(..., help="Experiment name"),
    run_name: str = typer.Argument(..., help="Run name"),
    policy_name: Optional[str] = typer.Argument("best", help="Policy to evaluate"),
    task: Optional[str] = typer.Option(None, help="Task to evaluate"),
):
    """Evaluates an experiment from a zip file."""
    print("Evaluating experiment...")
    runner = Runner.from_file(experiment_name, run_name, policy_name)
    for sweep in runner.experiment.sweeps:
        runner.evaluate(sweep, task=task)
    print("Evaluation finished :tada:")


if __name__ == "__main__":
    typer.run(main)
