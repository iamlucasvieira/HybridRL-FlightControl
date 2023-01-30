"""Project paths"""

import pathlib as pl
from dataclasses import dataclass
import os

# Project root
ROOT = pl.Path(__file__).resolve().parents[2]

# Data paths
SRC = ROOT / pl.Path("src")
DATA = SRC / pl.Path("data")
AIRCRAFT_DATA = DATA / pl.Path("aircraft")
MODELS = ROOT / pl.Path("models")
LOGS = ROOT / pl.Path("logs")
EXP = SRC / pl.Path("experiments")
FIGURES = ROOT / pl.Path("reports/figures")


@dataclass
class Path:
    """Class that contains all project paths."""
    root: pl.Path = ROOT
    src: pl.Path = SRC
    data: pl.Path = DATA
    aircraft_data: pl.Path = AIRCRAFT_DATA
    models: pl.Path = MODELS
    logs: pl.Path = LOGS
    exp: pl.Path = EXP
    figures: pl.Path = FIGURES
    paper_figures: pl.Path = pl.Path(
        "/Users/lucas/Documents/Thesis Writing/LiteratureReview/figures")  # Path to report figures


def set_wandb_path():
    os.environ["WANDB_DIR"] = Path.logs.as_posix()
