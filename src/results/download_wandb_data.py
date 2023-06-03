import pandas as pd
import wandb

from helpers.paths import Path


def download_summary_data(project, sweep=None, file_name=None, file_name_after=None):
    """Downloads the summary data from a project or sweep to a csv file."""
    api = wandb.Api()

    entity = "lucasv"
    if sweep is not None:
        runs = api.sweep(f"{entity}/{project}/{sweep}").runs
    else:
        runs = api.runs(f"{entity}/{project}")

    summary_list, config_list, name_list = [], [], []
    for run in runs:
        # .summary contains the output keys/values
        #  for metrics such as accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k, v in run.config.items() if not k.startswith("_")}
        )

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    # Make a dataframe from list of dicts, expanding dicts into columns
    summary_df = pd.DataFrame.from_records(summary_list)
    config_df = pd.DataFrame.from_records(config_list)
    name_df = pd.DataFrame({"name": name_list})
    runs_df = pd.concat([name_df, config_df, summary_df], axis=1)

    if file_name is None:
        file_name = f"{Path.data}/{project}"
        if sweep is not None:
            file_name += f"_{sweep}"
        if file_name_after is not None:
            file_name += f"_{file_name_after}"
        file_name += ".csv"

    runs_df.to_csv(file_name)


if __name__ == "__main__":
    download_summary_data("idhp-sac-hyperparams", sweep="050zos9u")
