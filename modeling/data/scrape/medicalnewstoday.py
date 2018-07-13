"""
This is a web scraping tool that has been setup to scrape content for the website

    Medical News Today - https://www.medicalnewstoday.com
"""
import click
import bs4
import itertools
import json
import os
import logging

from bs4 import BeautifulSoup

from .util import retry_get


logger = logging.getLogger(__name__)


@click.command()
@click.argument('start-page', type=int)
@click.argument('end-page', type=int)
@click.argument('filename', type=str)
def scrape_medicalnewstoday(start_page, end_page, filename):
    page_range = list(range(start_page, end_page+1))
    logger.info(f'Request to read medicalnewstoday for pages {page_range} to file {filename}')

    read_from_medicalnewstoday(page_range, filename)


@click.command()
@click.argument('filename-template')
@click.option('--threads', '-t', default=5, type=int)
def scrape_all_medicalnewstoday(filename_template, threads):
    pass


def read_from_medicalnewstoday(page_numbers, save_filename):
    """
    """
    results = {}
    for page_number in page_numbers:
        code, articles = get_articles_from_archive_page(page_number)
        if code < 200 or code >= 300:
            results[page_number] = (code, articles)
            continue

        for article in articles:
            code, text = read_article_text(article['link'])
            article['status_code'] = code
            article['text'] = text

        results[page_number] = articles

    with open(save_filename, 'w') as f:
        json.dump(results, f, indent=4)


def _get_articles_from_archive_page(page_number=None):
    """
    """
    base_url = 'https://www.medicalnewstoday.com'
    archive_url = f'{base_url}/archive/'
    if page_number is not None:
        archive_url = archive_url + str(page_number)

    try:
        status_code, content = retry_get(archive_url)

        if status_code < 200 or status_code >= 300:
            return status_code, []

        soup = BeautifulSoup(content, 'html5lib')

        article_list = soup.find('ul', class_='listing')
        articles = article_list.find_all('li')
    except Exception as e:
        print(f'Failed to get details from {archive_url} - {e}')
        return -1, []

    return status_code, [
        {
            'link': f'{base_url}{listing.a["href"]}',
            'title': listing.a['title'],
            'type': listing['class'][0],
            'span_class': listing.span['class'][0],
            'span_text': listing.span.text.strip(),
            'time': listing.find('span', class_='story_metadata').span.text
        } for listing in articles
    ]


def _read_article_text(url):
    """
    """
    try:
        status_code, content = retry_get(url)

        if status_code < 200 or status_code >= 300:
            return status_code, str(content)

        soup = BeautifulSoup(content, 'html5lib')
        article_tag = soup.find('div', class_='article_body')
        article_body = article_tag.find('div', itemprop='articleBody')

        tags_to_remove = itertools.chain(*[
            article_body.find_all('script'),
            article_body.find_all('div', class_='article_toc'),
            article_body.find_all('span', class_='imageWidgetWrapper'),
            article_body.find_all('div', class_='leaderboard'),
            article_body.find_all('img'),
            article_body.find_all('div', class_='related_inline'),
            article_body.find_all('div', class_='photobox_right'),
            article_body.find_all('div', class_='photobox_left')
        ])

        for tag in tags_to_remove:
            tag.extract()

        return status_code, article_body.get_text()
    except Exception as e:
        print(f'Failed to read article {url} - {e}')
        return -1, None

