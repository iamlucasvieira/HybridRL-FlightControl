import pandas as pd
import wandb

from helpers.paths import Path


class Wandb:
    """Class that handles download of data from wandb."""

    def __init__(self, project, sweep=None, file_name=None):
        """Initialize the class."""
        self.project = project
        self.sweep = sweep
        self.entity = "lucasv"
        self.file_name = file_name
    def get_file_name(self, file_name_after=None):
        file_name = self.file_name
        if file_name is None:
            file_name = f"{Path.data}/{self.project}"
            if self.sweep is not None:
                file_name += f"_{self.sweep}"
            if file_name_after is not None:
                file_name += f"_{file_name_after}"
            file_name += ".csv"
        return file_name
    def loop_through_runs(self, callback):
        api = wandb.Api()

        if self.sweep is not None:
            runs = api.sweep(f"{self.entity}/{self.project}/{self.sweep}").runs
        else:
            runs = api.runs(f"{self.entity}/{self.project}")
        idx = 1
        for run in runs:
            callback(run)
            print(f"Run {idx}/{len(runs)}")
            idx += 1

    def download_summary(self):
        summary_list, config_list, name_list = [], [], []

        def callback(run):
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

        self.loop_through_runs(callback)

        summary_df = pd.DataFrame.from_records(summary_list)
        config_df = pd.DataFrame.from_records(config_list)
        name_df = pd.DataFrame({"name": name_list})
        runs_df = pd.concat([name_df, config_df, summary_df], axis=1)

        runs_df.to_csv(self.get_file_name("summary"))

    def download_online(self):
        columns = [["online/phi", "online/phi_ref"],
                   ["online/theta", "online/theta_ref"],
                   ["online/beta", "online/beta_ref"]]
        step = ["online/step"]

        all_data = []
        def callback(run):
            run_data = []
            for column in columns:
                column_data = run.scan_history(keys=column + step)
                column_data_list = [dict(data, **{"name": run.name}) for data in column_data]
                run_data.append(pd.DataFrame.from_records(column_data_list))

            final_df = run_data[0]
            for df in run_data[1:]:
                final_df = pd.merge(final_df, df, on=["online/step", "name"], how="outer")
            all_data.append(final_df)
        self.loop_through_runs(callback)
        df = pd.concat(all_data)
        df.to_csv(self.get_file_name("online"))


if __name__ == "__main__":
    wandb_object = Wandb("exp3_fault", sweep="fp41w198")
    wandb_object.download_summary()
