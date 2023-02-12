# -*- coding: utf-8 -*-
import typer
from thesis_pilot.core import Experiment
from typing import Optional
import pathlib as pl
from helpers.paths import Path
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import IntPrompt, Confirm

app = typer.Typer()
console = Console()


def available_experiment_files():
    """Returns a list of available experiment files."""
    return [f.name for f in Path.exp.iterdir() if f.is_file() and f.suffix == ".yaml"]


@app.command()
def main(filename: Optional[str] = typer.Argument(None, help="Experiment file name"),
         filepath: Optional[pl.Path] = typer.Argument(Path.exp, help="Experiment file path"),
         offline: Optional[bool] = typer.Option(False, help="Run experiment offline"), ):
    """Runs an experiment from a config file."""
    if filename is None:
        table = Table(header_style="bold magenta")
        table.add_column("Id", justify="left", style="green", no_wrap=True)
        table.add_column("Experiment", justify="left", style="green")
        experiments = available_experiment_files()
        for idx, file in enumerate(experiments):
            table.add_row(str(idx + 1), file)

        panel = Panel(table, title="Available experiments to run", title_align="center", border_style="magenta",
                      expand=False)
        console.print(panel)

        if Confirm.ask("Do you want to run an experiment?"):
            idx = IntPrompt.ask("Select an experiment", choices=[str(i) for i in range(1, len(experiments) + 1)])
            filename = experiments[idx - 1]
        else:
            raise typer.Exit()
    filename = filename + ".yaml" if not filename.endswith(".yaml") else filename
    file = filepath / filename
    if not file.exists():
        print(f"File {file} does not exist.")
        raise typer.Abort()

    exp = Experiment(filename, file_path=filepath, offline=offline)
    exp.learn()


if __name__ == "__main__":
    typer.run(main)
