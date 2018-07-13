import logging
import click

from .medicalnewstoday import scrape_all_medicalnewstoday, scrape_medicalnewstoday


@click.group()
def cli():
    logging.basicConfig(level=logging.INFO)

cli.add_command(scrape_all_medicalnewstoday)
cli.add_command(scrape_medicalnewstoday)


if __name__ == '__main__':
    cli()
