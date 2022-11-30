# -*- coding: utf-8 -*-
import click
import os
from experiments import lti_citation
@click.group(context_settings={'show_default': True})
@click.option('--no-log', '-nl', default=True, is_flag=True, help='Enable logging.')
@click.pass_context
def main(ctx, no_log):
    """Main CLI entrypoint."""
    ctx.ensure_object(dict)

    ctx.obj['no_log'] = no_log


@main.command()
@click.option('--algo', '-a', default='SAC', help='Algorithm to use.')
@click.option('--env', '-e', default='citation', help='Environment to use.')
@click.option('--task', '-t', default='aoa', help='Task to use.')
@click.option('--seed', '-s', default=1, help='Random seed.')
@click.option('--dt', '-dt', default=0.1, help='Time step.')
@click.option('--episode-steps', '-es', default=100, help='Number of steps in an episode.')
@click.option('--global-steps', '-gs', default=1_000, help='Number of total learning steps.')
@click.option('--offline', '-off', default=False, is_flag=True, help='Disable wandb sync.')
@click.option('--verbose', '-v', default=1, help='Verbosity level.')
@click.option("--to-plot", "-tp", default=False, is_flag=True, help="Plot results.")
@click.pass_context
def learn(ctx, algo: str, env: str, task: str, seed: int, dt: float, episode_steps: int, global_steps: int,
          offline: bool, verbose: int, to_plot: bool):

    if offline:
        os.environ["WANDB_MODE"] = "offline"

    lti_citation.main(
        algorithm_name=algo,
        env_name=env,
        task=task,
        seed=seed,
        dt=dt,
        episode_steps=episode_steps,
        learning_steps=global_steps,
        verbose=verbose,
        TO_PLOT=to_plot,
    )

if __name__ == '__main__':
    main()
