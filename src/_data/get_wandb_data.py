from helpers.paths import Path
import wandb
import pandas as pd


def get_wandb_data(project_name):
    """Function that gathers the episode and learning _data from wandb."""

    def get_columns(_run, keys):
        """Return a dataframe with the passed_keys"""
        return _run.history(keys=keys)[keys]

    file_path_episode = Path.data / f"{project_name}_episode.csv"
    file_path_learning = Path.data / f"{project_name}_learning.csv"

    if file_path_episode.is_file() or file_path_learning.is_file():
        print(f"File {file_path_episode} or {file_path_learning} already exists")
    else:
        api = wandb.Api(timeout=29)
        runs = api.runs(f"lucasv/{project_name}")
        df_all_episode, df_all_learning = pd.DataFrame(), pd.DataFrame()
        for run in runs:
            # Episode _data
            episode_data_list = [
                get_columns(run, ["episode_step", "tracking_error"]),
                get_columns(run, ["episode_step", "state", "reference"]),
                get_columns(run, ["episode_step", "reward"]),
                get_columns(run, ["episode_step", "action"]),
            ]

            df_episode = pd.concat(
                [df.set_index("episode_step") for df in episode_data_list], axis=1
            ).reset_index()

            # Training _data
            learning_data_list = [
                get_columns(run, ["rollout/ep_len_mean", "global_step"]),
                get_columns(run, ["rollout/ep_rew_mean", "global_step"]),
            ]

            df_learning = pd.concat(
                [df.set_index("global_step") for df in learning_data_list], axis=1
            ).reset_index()

            # Add information to _data
            for df in [df_episode, df_learning]:
                df["algorithm"] = run.config["algorithm"].upper()
                df["reward_type"] = run.config["reward_type"]
                df["task_type"] = run.config["task_type"]
                df["run"] = run.name
                if "obs" in run.config:
                    df["observation"] = run.config["obs"]

            df_all_episode = pd.concat([df_all_episode, df_episode])
            df_all_learning = pd.concat([df_all_learning, df_learning])

        df_all_episode.to_csv(file_path_episode)
        df_all_learning.to_csv(file_path_learning)

        print(f"Saved {file_path_episode} and {file_path_learning}")


get_wandb_data("Tasks")
