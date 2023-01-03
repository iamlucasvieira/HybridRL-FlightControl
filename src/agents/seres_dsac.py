from agents.dsac.dsac_agent import DSACAgent

from helpers.config import ConfigLinearAircraft
from models.aircraft_environment import AircraftEnv
import numpy as np
import wandb
import torch
import random

def torchify(x: np.ndarray) -> torch.Tensor:
    """Turn a single dimensional numpy array to a tensor of size [B, N]"""
    return torch.tensor(x).float().unsqueeze(0)


class DSAC:
    agent_config = {
        "n_hidden_layers": 2,
        "n_hidden_units": 64,
        "lr": 4.4e-4,
        "gamma": 0.99,
        "batch_size": 256,
        "buffer_size": 500_000,
        "polyak_step_size": 0.995,
        "update_every": 1,
        "lambda_t": 400.0,
        "lambda_s": 400.0,
        "n_taus": 8,
        "n_cos": 64,
        "n_taus_exp": 8,  # number of taus used for calculating the expectation
        "risk_distortion": "wang",  # Options: "wang", "", "", ""
        "risk_measure": -0.5
    }

    def __init__(self, policy, env, verbose=0, seed=None, **kwargs):
        self.policy = policy
        self.env = env
        self.verbose = verbose

        if seed is not None:
            # Seed python RNG
            random.seed(seed)
            # Seed numpy RNG
            np.random.seed(seed)

        self.agent = DSACAgent(
            config=DSAC.agent_config,
            device="cpu",
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
        )

    def learn(self, total_timesteps: int = 1000, callback=None, log_interval: int = 1, tb_log_name="run",
              progress_bar=False, **kwargs):
        """Train the agent through n_steps."""

        n_steps = total_timesteps

        # Reward stats:
        rewards, episode_length = [], []
        mean_rewards, mean_length = [], []
        total_steps = 0

        # Iterate through episodes:
        while total_steps < n_steps:
            total_reward, steps = self.train_single_episode()
            total_steps += steps

            # Save rewards and lengths:
            rewards.append(total_reward)
            episode_length.append(steps)

            if len(rewards) % log_interval == 0:
                mean_rewards.append(np.mean(rewards[-log_interval:]))
                mean_length.append(np.mean(episode_length[-log_interval:]))

                if wandb.run is not None:
                    wandb.log({"rollout/ep_rew_mean": mean_rewards[-1],
                               "global_step": total_steps})

                    wandb.log({"rollout/ep_len_mean": mean_length[-1],
                               "global_step": total_steps})

            self.print(f"Total steps: {total_steps}, reward: {total_reward}")
        self.print(f"Mean reward: {mean_rewards}")

    def train_single_episode(self):
        """Returns the (episode data object, end-of-episode return, wall-clock training time)"""

        env = self.env
        agent = self.agent

        # Reset environment
        state = env.reset()
        total_reward = 0
        n_steps = 0

        while True:

            # steps counter
            n_steps += 1

            # callback:
            agent.every_sample_update_callback()

            # Choose action
            action = agent.act(state)

            # Step the environment
            next_state, reward, done, info = env.step(action)

            # Fetch the variances of the critics:
            try:
                var_1 = agent.Z1.get_variance(
                    s=torchify(state), a=torchify(action), n_taus=16
                )
                var_2 = agent.Z1.get_variance(
                    s=torchify(state), a=torchify(action), n_taus=16
                )
            except AttributeError:
                var_1 = 0.0
                var_2 = 0.0

            # Accumulate reward
            total_reward += reward

            # Update the agent
            agent.update(state, action, reward, next_state, done)

            # Swap the states
            state = next_state

            # Break when the episode is done
            if done:
                break

        return total_reward, n_steps

    def print(self, text):
        if self.verbose > 0:
            print(text)

    def predict(self, observation, state=None, mask=None, deterministic=False):
        return self.agent.act(observation), observation

    def save(self, file_path):
        self.agent.save_to_file(file_path)

    def load(self, file_path):
        self.agent = self.agent.load(file_path=file_path,
                                     config=DSAC.agent_config,
                                     device="cpu",
                                     obs_dim=self.env.observation_space.shape[0],
                                     action_dim=self.env.action_space.shape[0], )


def main():
    pass


if __name__ == "__main__":
    main()
