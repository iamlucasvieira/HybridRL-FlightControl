"""Module that defines helper functions for wandb."""
import wandb
import torch


def evaluate(agent, env, n_times=1):
    """Run the experiment n times."""

    for _ in range(n_times):
        obs = env.reset()
        done = False
        steps = 0
        while not done:
            action, _states = agent.predict(obs, deterministic=True)

            # Transform action into numpy if it is a tensor
            if isinstance(action, torch.Tensor):
                action = action.detach().numpy()

            obs, reward, done, info = env.step(action)

            env.render()

            if wandb.run is not None:
                wandb.log({"reward": reward,
                           "episode_step": steps})
                wandb.log({"reference": env.reference[-1],
                           "state": env.track[-1],
                           "episode_step": steps})
                wandb.log({"action": action,
                           "episode_step": steps, })
                wandb.log({"tracking_error": env.sq_error[-1],
                           "episode_step": steps})

            steps += 1
        print(f"finished at {steps - 1}")
