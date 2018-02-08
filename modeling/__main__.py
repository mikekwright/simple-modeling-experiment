import os
import click
import json
import sys
import logging

from .util import JsonLoader


## Dynamic Data for JSON loading
from .data import *
from .evaluator import *
from .training import *

from .models.knn_model import KNNModel


def get_class_from_name(name):
    try:
        return getattr(sys.modules[__name__], name)
    except:
        raise Exception(f'Could not find a matching class for {name}')


def setup_logging(debug, output_path=None):
    handlers = [logging.StreamHandler(stream=sys.stdout)]

    if output_path:
        handlers.append(logging.FileHandler(os.path.join(output_path, 'details.log')))

    level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(handlers=handlers, level=level,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")


@click.group()
@click.option('--debug/--no-debug', default=False)
@click.option('--output', default=None)
@click.pass_context
def cli(ctx, debug, output=None):
    click.echo('Debug mode is %s' % ('on' if debug else 'off'))
    setup_logging(debug, output)
    ctx.obj['OUTPUT'] = output


# @cli.command('report')
# @click.argument('report', type=click.Choice(['feature_distance']))
# @click.option('')


@cli.command('train')
@click.argument('config_filename')
@click.pass_context
def run_training(ctx, config_filename):
    output = ctx.obj['OUTPUT']
    loader = JsonLoader(class_lookup_func=get_class_from_name)

    with open(config_filename, 'r') as config_file:
        train = loader.initialize_json(config_file.read())

    train()
    if output is not None:
        train.store_results(output)


if __name__ == '__main__':
    cli(obj={})
