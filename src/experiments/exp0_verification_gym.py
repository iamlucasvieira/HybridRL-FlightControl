"""Experiments to verify the algorithms on gym environments."""
import gymnasium as gym
from agents import SAC, DSAC
from agents.config.config_sac import ConfigSACLearn
from helpers.paths import Path
from agents.callbacks import SACCallback, TensorboardCallback

import wandb

run = wandb.init(project="hrl-fc-verification",
                 config={"agent": "dsac"})

env = gym.make("Pendulum-v1", render_mode="scii",)

sac = DSAC(env, log_dir=str(Path().logs), device="cpu")

sac.learn(
    total_steps=100000,
    callback=[SACCallback(), TensorboardCallback()],
    log_interval=10,
    run_name=run.name,
)

observation, info = env.reset(seed=51)


for time in range(1000):
    action = sac.predict(observation)  # this is where you would insert your policy
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"Time: {time},  Reward: {reward}")

    if terminated or truncated:
        print("END")
        break

    wandb.log(
        {"eval/observation": observation, "eval/reward": reward, "eval/time": time}
    )


env.close()
wandb.finish()
