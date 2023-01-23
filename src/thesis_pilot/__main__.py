# -*- coding: utf-8 -*-
import click
import os
from experiments.core import Sweep, Experiment


@click.group(context_settings={'show_default': True})
@click.option('--no-log', '-nl', default=True, is_flag=True, help='Enable logging.')
@click.pass_context
def main(ctx, no_log):
    """Main CLI entrypoint."""
    ctx.ensure_object(dict)

    ctx.obj['no_log'] = no_log


class ConfigCommands(click.core.Command):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params.insert(0, click.core.Option(('--algo', '-a'), default="SAC", help='Algorithm to use.'))
        self.params.insert(0, click.core.Option(('--env', '-e'), default="citation", help='Environment to use.'))
        self.params.insert(0, click.core.Option(('--task', '-t'), default="q_sin", help='Task to use.'))
        self.params.insert(0, click.core.Option(('--seed', '-s'), default=None, help='Seed to use.'))
        self.params.insert(0, click.core.Option(('--dt', '-dt'), default=0.1, help='Time step to use.'))
        self.params.insert(0,
                           click.core.Option(('--episode-steps', '-es'), default=100, help='Number of steps to use.'))
        self.params.insert(0, click.core.Option(('--learning-steps', '-ls'), default=1000,
                                                help='Number of learning steps to use.'))
        self.params.insert(0, click.core.Option(('--offline', '-o'), default=False, is_flag=True,
                                                help='Whether to run offline.'))
        self.params.insert(0, click.core.Option(('--project-name', '-pn'), default="", help='Name of the project.'))
        self.params.insert(0, click.core.Option(('--name', '-n'), default="", help='Name of the run.'))
        self.params.insert(0, click.core.Option(('--verbose', '-v'), default=2, help='Verbosity level.'))
        self.params.insert(0, click.core.Option(('--config', '-c'), default="sp", help='Configuration to use.'))
        self.params.insert(0, click.core.Option(('--evaluate', '-ev'), default=1,
                                                help='Number of evaluations to perform.'))
        self.params.insert(0, click.core.Option(('--tags', '-tg'), default=None, help='Tags to use.'))
        self.params.insert(0, click.core.Option(('--reward-type', '-rt'), default="sq_error",
                                                help='Type of the reward function'))
        self.params.insert(0, click.core.Option(('--observation-type', '-ot'), default="error", help='Type of the observation'))


@main.command()
@click.argument('name')
@click.option('--offline', '-o', default=None, is_flag=True, help='Whether to run offline.')
def experiment(name: str, offline: bool):
    """Run an experiment."""

    exp_args = {}
    if offline is not None:
        exp_args = {'offline': offline}
    exp = Experiment(name, *exp_args)
    exp.learn()


@main.command(cls=ConfigCommands)
@click.pass_context
def learn(ctx, **kwargs):
    """Learn a task."""
    exp = Sweep(algorithm_name=kwargs['algo'],
                env_name=kwargs['env'],
                configuration=kwargs['config'],
                task_name=kwargs['task'],
                seed=kwargs['seed'],
                dt=kwargs['dt'],
                episode_steps=kwargs['episode_steps'],
                learning_steps=kwargs['learning_steps'],
                verbose=kwargs['verbose'],
                offline=kwargs['offline'],
                project_name=kwargs['project_name'],
                reward_type=kwargs['reward_type'],
                observation_type=kwargs['observation_type'],
                evaluate=kwargs['evaluate'],
                )

    exp.learn(name=kwargs['name'], tags=kwargs['tags'])


if __name__ == '__main__':
    main()
