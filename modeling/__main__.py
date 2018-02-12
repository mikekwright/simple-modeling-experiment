import os
import click
import json
import sys
import logging

from jsoncomment import JsonComment
from jsonmerge import merge

from .util import JsonLoader


## Dynamic Data for JSON loading
from .data import *
from .evaluator import *
from .training import *
from .transform import *
from .models import *

from .models.knn import *


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
    if output:
        os.makedirs(output, exist_ok=True)

    setup_logging(debug, output)
    ctx.obj['OUTPUT'] = output


# @cli.command('report')
# @click.argument('report', type=click.Choice(['feature_distance']))
# @click.option('')


@cli.command('train')
@click.argument('configs', nargs=-1)
@click.pass_context
def run_training(ctx, configs):
    """
    Takes in a list of json configs, combines them and runs a training using the config contents
    """
    if not configs:
        click.echo(f'You must supply at least one config')
        return

    config_contents = combine_configs(configs)
    output = ctx.obj['OUTPUT']
    loader = JsonLoader(class_lookup_func=get_class_from_name)

    train = loader.initialize_json(config_contents)
    train()

    if output is not None:
        with open(os.path.join(output, 'train_config.json'), 'w', encoding='utf-8') as config_file:
            json.dump(json.loads(config_contents), config_file, indent=4, ensure_ascii=False)
        train.store_results(output)


@cli.command('combine')
@click.argument('configs', nargs=-1)
@click.pass_context
def combine_command(ctx, configs):
    """
    Takes in a list of json comfigs, combines them and prints out the results
    """
    config_contents = json.loads(combine_configs(configs))

    output = ctx.obj['OUTPUT']
    if output:
        with open(output, 'w', encoding='utf-8') as out_file:
            json.dump(config_contents, out_file, indent=4, ensure_ascii=False)
    else:
        print(json.dumps(config_contents, indent=4, ensure_ascii=False))


def combine_configs(configs):
    contents = {}
    parser = JsonComment(json)

    for config_filename in configs:
        with open(config_filename, 'r', encoding='utf-8') as cf:
            file_contents = parser.load(cf)

        contents = merge(contents, file_contents)

    return json.dumps(contents, indent=4)



if __name__ == '__main__':
    cli(obj={})
