"""Utility functions for visualization."""
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib import font_manager
import matplotlib.lines as mlines

from helpers.paths import Path


def make_defaults():
    """Set up default matplotlib settings."""
    sns.set()
    # sns.set_context("paper")
    sns.set_palette(px.colors.qualitative.D3)

    font_path = '/Users/lucas/Library/Fonts/FiraSans-Regular.ttf'  # Your font path goes here
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = prop.get_name()


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


def wandb2df(df_start, step_column):
    """Gets a df from wandb csv and returns a df with the columns as variables."""
    df = df_start.melt(id_vars=step_column, var_name='id_var', value_name='value')
    df[['id', 'var']] = df['id_var'].str.split(' - ', expand=True)
    df = df.pivot_table(index=[step_column, 'id'], columns='var', values='value').reset_index()
    for col in df.columns:
        if '/' in col:
            new_col_name = col.split('/')[-1]
            df.rename(columns={col: new_col_name}, inplace=True)
    return df


def get_track_and_states_df(df_summary, df_track_initial, df_states_initial, task=None):
    """Gets the track and states df and adjust them"""

    if task is not None:
        runs = df_summary[df_summary['task_train'] == task]["name"].values
        df_track = df_track_initial[df_track_initial['id'].isin(runs)]
        df_states = df_states_initial[df_states_initial['id'].isin(runs)]
    else:
        df_track = df_track_initial
        df_states = df_states_initial
    interest_columns = ["step", "theta", "theta_ref", "phi", "phi_ref", "beta",
                        "beta_ref", "de", "da", "dr"]
    df_track = df_track[interest_columns]
    df_track = df_track.apply(lambda x: np.rad2deg(x) if x.name not in ['step'] else x)
    df_track['step'] *= 0.01

    interest_columns = ["step", "V", "he", "p", "q", "r", "alpha"]
    df_states = df_states[interest_columns]
    df_states = df_states.apply(lambda x: np.rad2deg(x) if x.name not in ['step', 'V', 'he'] else x)
    df_states['step'] *= 0.01

    return df_track, df_states


def plot_states(df_track, df_states, with_legend=True):
    dpi = 200

    fig, axes = plt.subplots(6, 2, sharex=True, figsize=(5, 7), dpi=dpi)

    state_color = "tab:blue"  # Assign consistent color to state lines
    ref_color = "tab:gray"  # Assign consistent color to reference lines

    # Plot with consistent color scheme and legend
    def plot_data(data, x, y, ax, ylabel, ref_data=None, ref_y=None):
        sns.lineplot(data=data, x=x, y=y, ax=ax, color=state_color)
        if ref_data is not None and ref_y is not None:
            sns.lineplot(data=ref_data, x=x, y=ref_y, ax=ax, color=ref_color, linestyle="dashed")
        ax.set_ylabel(ylabel)
        ax.yaxis.set_label_coords(-0.2, 0.5)
        # ax.set_xlabel("Time [s]", fontsize=14)  # Label x-axis for each subplot
        # ax.legend(fontsize=12)  # Adjust legend size
        # ax.tick_params(axis='both', which='major', labelsize=12)  # Adjust tick label size

    # Call plot_data for each subplot
    plot_data(df_states, 'step', 'q', axes[0, 0], r"$q$ [deg/s]")
    plot_data(df_states, 'step', 'p', axes[0, 1], r"$p$ [deg/s]")
    plot_data(df_states, 'step', 'alpha', axes[1, 0], r"$\alpha$ [deg]")
    plot_data(df_states, 'step', 'r', axes[1, 1], r"$r$ [deg/s]")
    plot_data(df_track, 'step', 'theta', axes[2, 0], r"$\theta$ [deg]", df_track, 'theta_ref')
    plot_data(df_track, 'step', 'phi', axes[2, 1], r"$\phi$ [deg]", df_track, 'phi_ref')
    plot_data(df_states, 'step', 'V', axes[3, 0], r"$V$ [m/s]")
    plot_data(df_track, 'step', 'beta', axes[3, 1], r"$\beta$ [deg]", df_track, 'beta_ref')
    plot_data(df_states, 'step', 'he', axes[4, 0], r"$h$ [m]")
    plot_data(df_track, 'step', 'da', axes[4, 1], r"$\delta_{a}$ [deg]")
    plot_data(df_track, 'step', 'de', axes[5, 0], r"$\delta_{e}$ [deg]")
    plot_data(df_track, 'step', 'dr', axes[5, 1], r"$\delta_{r}$ [deg]")

    axes[5, 0].set_xlabel("Time [s]")
    axes[5, 1].set_xlabel("Time [s]")

    if with_legend:
        state_line = mlines.Line2D([], [], color=state_color, label='State')
        ref_line = mlines.Line2D([], [], color=ref_color, linestyle="dashed", label='Reference')
        fig.legend(handles=[state_line, ref_line], loc='upper center', ncol=2, fontsize=12)

    plt.tight_layout(pad=0.2, rect=[0, 0, 1, 0.94])  # adjust the rectangle in which to fit the subplots

    return fig, axes


Style = namedtuple("style", ["font", "width"])

defaults = Style(
    font=dict(
        family="Fira Sans, Helvetica, Arial, sans-serif", color="#333333", size=20
    ),
    width=500,  # Pixels
)
