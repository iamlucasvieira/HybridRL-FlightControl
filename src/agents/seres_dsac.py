from agents.dsac.dsac_agent import DSACAgent

from helpers.config import ConfigLinearAircraft
from models.aircraft_environment import AircraftEnv
import numpy as np
from utils import torchify
import wandb


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

    def __init__(self, env, config, project_name='DASC'):

        self.env = env
        self.config = config
        self.project_name = project_name
        self.agent = DSACAgent(
            config=DSAC.agent_config,
            device="cpu",
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
        )

    def train(self, log_freq: int = 2):
        """Train the agent through n_steps."""

        config = self.config
        n_steps = config.learning_steps

        # Reward stats:
        rewards = []
        mean_rewards = []
        total_steps = 0

        run = wandb.init(
            project=self.project_name,
            config=config.dict(),
        )

        # Iterate through episodes:
        while total_steps < n_steps:
            total_reward, steps = self.train_single_episode()
            total_steps += steps

            # Save rewards:
            rewards.append(total_reward)

            if len(rewards) % log_freq == 0:
                mean_rewards.append(np.mean(rewards[-log_freq:]))
                wandb.log({"rollout/ep_rew_mean": mean_rewards[-1],
                           "global_step": total_steps})
            print(f"Total steps: {total_steps}, reward: {total_reward}")
        print(f"Mean reward: {mean_rewards}")

        run.finish()

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


def main():
    config = ConfigLinearAircraft()
    env = AircraftEnv(config)

    dsac = DSAC(env, config)
    dsac.train(n_steps=1000)


if __name__ == "__main__":
    main()
