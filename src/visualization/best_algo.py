import pandas as pd
from helpers.paths import Path

file_path = Path.data / "best_algo.csv"


def get_wandb_data():
    """Downlaod the wandb data for the best_algo project"""
    import wandb
    api = wandb.Api()

    runs = api.runs("lucasv/best_algo_v2")
    df = pd.DataFrame()

    for run in runs:
        history = run.history()
        for k, v in run.config.items():
            if not k.startswith('_'):
                history[k] = v
        history['run'] = run.name

        df = pd.concat([df, history])

    df.to_csv(file_path)

def columns_without_nan(df, column):
    return df[~df[column].isna()]

# get_wandb_data()
df = pd.read_csv(file_path)
print(1)
import matplotlib.pyplot as plt

for run in df['run'].unique():
    df_run = df[df.run == run]

    df_clean = columns_without_nan(df_run, 'rollout/ep_len_mean')
    plt.plot(df['global_step'], df['rollout/ep_len_mean'])
plt.show()
