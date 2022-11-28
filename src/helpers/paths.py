"""Project paths"""

import pathlib as pl
from dataclasses import dataclass

# Project root
ROOT = pl.Path(__file__).resolve().parents[2]

# Data paths
SRC = ROOT / pl.Path("src")
DATA = SRC / pl.Path("data")
AIRCRAFT_DATA = DATA / pl.Path("aircraft")
MODELS = ROOT / pl.Path("models")
LOGS = ROOT / pl.Path("logs")

@dataclass
class Path:
    """Class that contains all project paths."""
    root: pl.Path = ROOT
    src: pl.Path = SRC
    data: pl.Path = DATA
    aircraft_data: pl.Path = AIRCRAFT_DATA
    models: pl.Path = MODELS
    logs: pl.Path = LOGS
