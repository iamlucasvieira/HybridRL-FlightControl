"""Module that defines  the logger for the agents."""
from helpers.paths import Path
from typing import Optional, Any
import pathlib as pl
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from rich.table import Table
from hrl_fc.console import console


class Logger:
    """Logger class."""

    def __init__(self, log_dir: Optional[str], run_name: str, verbose: int) -> None:
        """Initialize the logger."""
        self.log_dir = log_dir
        self.verbose = verbose
        self.data = defaultdict(float)
        log_dir = Path.logs if log_dir is None else pl.Path(log_dir)
        self.path = log_dir / run_name

    def record(self, key: str, value: Any) -> None:
        self.data[key] = value

    def dump(self, step: int = 0):
        """Dump the data to the log directory."""
        if self.verbose > 0:
            self.print_table(step)
        # self.print(self.data)
        self.data.clear()

    def print(self, message) -> None:
        """Print a message."""
        if self.verbose > 0:
            print(message)

    def print_table(self, steps):
        """Print the table."""
        table = Table(title=f"Step {steps}")
        table.add_column("Key")
        table.add_column("Value")
        for key, value in self.data.items():
            table.add_row(key, str(value))
        console.print(table)
