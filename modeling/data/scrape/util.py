"""
This module contains common tools for web scraping.
"""
import requests


def retry_get(url, retry_count=3):
    attempt_count = 0
    while True:
        try:
            attempt_count += 1
            result = requests.get(url)
            code = result.status_code
            content = result.content

            if code >= 200 and code < 300:
                return code, content

            if attempt_count > retry_count:
                return code, content

        except Exception as e:
            print(f'Url {url} failed {attempt_count} times - last error {e}')
            if attempt_count > retry_count:
                return -1, None
