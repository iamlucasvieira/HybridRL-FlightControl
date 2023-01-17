from helpers.paths import Path
import wandb
import pandas as pd

project_name = "best_reward"
file_path = Path.data / f"{project_name}.csv"
file_path_rollout = Path.data / f"{project_name}_rollout.csv"

if file_path.is_file():
    print(f"File {file_path} already exists")
else:
    api = wandb.Api(timeout=19)
    runs = api.runs(f"lucasv/{project_name}")
    df_tracking, df_rollout = pd.DataFrame(), pd.DataFrame()
    for run in runs:
        # Tracking data
        history = run.history(keys=["episode_step", "tracking_error"])

        # Rollout data
        ep_len = run.history(keys=["rollout/ep_len_mean"])
        ep_rew = run.history(keys=["rollout/ep_rew_mean"])
        rollout = pd.merge(ep_len, ep_rew, on="_step", how="outer")

        # Add information to data
        for df in [history, rollout]:
            df["algorithm"] = run.config["algorithm"].upper()
            df["reward_type"] = run.config["reward_type"]
            df["run"] = run.name

        df_tracking = pd.concat([df_tracking, history])
        df_rollout = pd.concat([df_rollout, rollout])

    df_tracking.to_csv(file_path)
    df_rollout.to_csv(file_path_rollout)

    print(f"Saved {file_path}")
