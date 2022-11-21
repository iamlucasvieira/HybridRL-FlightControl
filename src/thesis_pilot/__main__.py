# -*- coding: utf-8 -*-
import click


@click.group()
@click.option('--no-log', '-nl', default=True, is_flag=True, help='Enable logging')
@click.pass_context
def main(ctx, no_log):
    """Main CLI entrypoint."""
    ctx.ensure_object(dict)

    ctx.obj['no_log'] = no_log


@main.command()
@click.pass_context
def sac(ctx):
    print(ctx.obj['no_log'])


if __name__ == '__main__':
    main()
