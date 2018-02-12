import os

from typing import Any

from .base import BaseStep


class StrToNumber(BaseStep):
    def __init__(self, field_names: str, result_names: str = None):
        super().__init__(field_names, result_names)

    def _handle_call(self, field: str, value: Any) -> float:
        return float(value)

    def store_results(self, directory: str):
        pass
