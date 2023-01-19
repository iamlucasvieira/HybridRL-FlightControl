"""Utility functions for visualization."""

import pandas as pd
import numpy as np


def make_smooth(_df, step=100, on=None, columns=[], info=[], group="run"):
    """Makes columns of the dataframe smooth by averaging over moving steps.

    Args:
        _df: Dataframe to smooth.
        step: Number of steps to average over.
        on: Column to count the steps on.
        columns: Columns to smooth.
        info: Columns with data about the group.
        group: Column to group by.
        """
    start = _df[on].min()
    stop = _df[on].max() + start
    all_data = []

    for run_name in _df[group].unique():
        # information about the run
        info_dict = {}
        for info_item in info:
            info_dict[info_item] = _df.loc[_df[group] == run_name, info_item].unique()[0]

        # Data from the run
        for t in np.arange(start, stop, step):
            mask = (_df[group] == run_name) & ((_df[on] >= t) & (_df[on] < t + step))
            run_slice = _df.loc[mask, columns]

            data = {"run": run_name, on: t, **info_dict}

            for column in columns:
                data[column] = run_slice[column].mean()

            all_data.append(data)

    return pd.DataFrame.from_records(all_data)
