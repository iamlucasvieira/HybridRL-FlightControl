"""Experiment to learn using wandb while using the citation model."""

import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor

import wandb
from helpers.misc import get_name
from helpers.paths import Path
from models.aircraft_environment import AircraftEnv
from helpers.tracking import SaveOnBestTrainingRewardCallback
from helpers.config import ConfigLinearAircraft
import pathlib as pl

config = ConfigLinearAircraft(
    algorithm="SAC",
    dt=0.1,
    episode_steps=1_000,
    learning_steps=5_000,
    task="aoa_sin",
)

TRAIN = True
env = AircraftEnv(config)

# name = get_name([config.env_name, config.algorithm])
project_name = f"{config.env_name}-{config.algorithm} v2"
MODELS_PATH = Path.models / project_name
LOGS_PATH = Path.logs / project_name

if TRAIN:

    run = wandb.init(
        project=project_name,
        config=config.asdict,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        save_code=True,  # optional
    )

    # Create directories
    pl.Path.mkdir(MODELS_PATH / run.name, parents=True, exist_ok=True)
    pl.Path.mkdir(LOGS_PATH, parents=True, exist_ok=True)

    env = Monitor(env, filename=f"{MODELS_PATH}/{run.name}")
    # wandb.tensorboard.patch(root_logdir=f"{LOGS_PATH}/{run.name}")

    # name = get_name([config.env_name, config.algorithm])

    run = wandb.init(
        project=project_name,
        config=config.asdict,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        save_code=True,  # optional
    )

    # Create directories
    pl.Path.mkdir(MODELS_PATH / run.name, parents=True, exist_ok=True)
    pl.Path.mkdir(LOGS_PATH, parents=True, exist_ok=True)

    env = Monitor(env, filename=f"{MODELS_PATH}/{run.name}")
    # wandb.tensorboard.patch(root_logdir=f"{LOGS_PATH}/{run.name}")

    # Define the callbacks for the training
    # best_model_callback = SaveOnBestTrainingRewardCallback(check_freq=config.episode_steps * 2,
    #                                                        log_dir=MODELS_PATH / run.name)
    wandb_callback = WandbCallback(
        model_save_freq=100,
        # gradient_save_freq=config.episode_steps,
        model_save_path=f"{MODELS_PATH / run.name}",
        verbose=2)

    # Creates the model algorithm
    model = SAC("MlpPolicy", env, verbose=2,
                tensorboard_log=LOGS_PATH)

    model.learn(total_timesteps=config.learning_steps,
                callback=[wandb_callback],#, best_model_callback],
                log_interval=1,
                tb_log_name=run.name,
                progress_bar=False)

    # Replace previous latest-model with the new model
    model.save(f"{MODELS_PATH}/latest-model")
else:
    model_name = "olive-sun-4"
    model = SAC.load(Path.models / project_name / model_name / "model.zip")

for _ in range(1):
    obs = env.reset()

    for i in range(config.episode_steps):
        action, _states = model.predict(obs, deterministic=True)

        obs, reward, done, info = env.step(action)
        env.render()

        if wandb.run is not None:
            wandb.log({f"reward": reward})
            wandb.log({f"reference": env.reference[-1]})
            wandb.log({f"state": env.track[-1]})


        if done:
            print(f"finished at {i}")
            break

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(env.track)
    ax[0].plot(env.reference, '--')

    ax[1].plot(env.actions)

    plt.show()

if wandb.run is not None:
    run.finish()

