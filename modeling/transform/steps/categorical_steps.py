import os
import json

from .base import BaseStep


class CategoryNumeric(BaseStep):
    def __init__(self, field_names, result_names=None, category_values=None):
        """
        """
        super().__init__(field_names, result_names)
        self._category_values = category_values or {}

    def two_pass(self):
        for field in self.field_names:
            if field not in self._category_values:
                return True

        return False

    def prep_pass(self, train_data):
        for field in self.field_names:
            if field in self._category_values:
                continue

            values = set([t[0][field] for t in train_data])
            self._category_values[field] = list(values)

    def _handle_call(self, field, value):
        return self._category_values[field].index(value)

    def store_results(self, directory):
        config = {
            'field_names': self.field_names,
            'result_names': self.result_names,
            'categories': self._category_values
        }
        with open(os.path.join(directory, 'categorical.json'), 'w', encoding='utf-8') as cf:
            json.dump(config, cf, indent=4, ensure_ascii=False)


class MultiCategorical(BaseStep):
    def __init__(self, field_names, result_names=None, category_values=None):
        """
        """
        super().__init__(field_names, result_names)
        self._category_values = category_values or {}

    def two_pass(self):
        for field in self.field_names:
            if field not in self._category_values:
                return True

        return False

    def prep_pass(self, train_data):
        for field in self.field_names:
            if field in self._category_values:
                continue

            values = set([t[0][field] for t in train_data])
            self._category_values[field] = list(values)

    def _handle_call(self, field, value):
        return [1 if value == v else 0 for v in self._category_values[field]]

    def store_results(self, directory):
        config = {
            'field_names': self.field_names,
            'result_names': self.result_names,
            'categories': self._category_values
        }
        with open(os.path.join(directory, 'categorical.json'), 'w', encoding='utf-8') as cf:
            json.dump(config, cf, indent=4, ensure_ascii=False)
