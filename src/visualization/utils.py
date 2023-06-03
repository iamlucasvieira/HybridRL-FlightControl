"""Utility functions for visualization."""
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

from helpers.paths import Path


def make_defaults():
    """Set up default matplotlib settings."""
    sns.set()
    # sns.set_context("paper")
    sns.set_palette(px.colors.qualitative.D3)


def make_smooth(_df, step=100, on=None, columns=[], info=[], group="run"):
    """Makes columns of the dataframe smooth by averaging over moving steps.

    Args:
        _df: Dataframe to smooth.
        step: Number of steps to average over.
        on: Column to count the steps on.
        columns: Columns to smooth.
        info: Columns with _data about the group.
        group: Column to group by.
    """
    start = _df[on].min()
    stop = _df[on].max() + start
    all_data = []

    for run_name in _df[group].unique():
        # information about the run
        info_dict = {}
        for info_item in info:
            info_dict[info_item] = _df.loc[_df[group] == run_name, info_item].unique()[
                0
            ]

        # Data from the run
        for t in np.arange(start, stop, step):
            mask = (_df[group] == run_name) & ((_df[on] >= t) & (_df[on] < t + step))
            run_slice = _df.loc[mask, columns]

            data = {"run": run_name, on: t, **info_dict}

            for column in columns:
                data[column] = run_slice[column].mean()

            all_data.append(data)

    return pd.DataFrame.from_records(all_data)


def save_pgf(fig, name, path=None, tight_layout=True, padding=0.5, w=3, h=3.5):
    """Sets matplotlib to pgf and saves figure."""
    if path is None:
        path = Path.figures
    elif path == "paper":
        path = Path.paper_figures
    plt.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,  # use inline math for ticks
            "pgf.rcfonts": False,  # don't setup fonts from rc parameters
        }
    )

    if tight_layout:
        fig.tight_layout(pad=padding)
    # fig.set_size_inches(w=w, h=h)
    fig.savefig(path / name)
    # w=6.202 inches
    # reset plt params
    # plt.rcParams.update(plt.rcParamsDefault)
    # make_defaults()


Style = namedtuple("style", ["font", "width"])

defaults = Style(
    font=dict(
        family="Fira Sans, Helvetica, Arial, sans-serif",
        color="#333333",
        size=20),
    width=500,  # Pixels

)
