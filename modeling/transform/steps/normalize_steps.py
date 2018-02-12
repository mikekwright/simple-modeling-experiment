import os
import json

from .base import BaseStep


class Normalize(BaseStep):
    def __init__(self, field_names, result_names=None, min_max_values=None):
        super().__init__(field_names, result_names)
        self._min_max_values = min_max_values or {}

    def two_pass(self):
        for field in self.field_names:
            if field not in self._min_max_values:
                return True

        return False

    def prep_pass(self, train_data):
        for field in self.field_names:
            values = [float(t[0][field]) for t in train_data]
            self._min_max_values[field] = (min(values), max(values))

    def _handle_call(self, field, value):
        min_value, max_value = self._min_max_values[field]
        return (value - min_value) / (max_value - min_value)

    def store_results(self, directory):
        os.makedirs(directory, exist_ok=True)
        config = {
            'features_values': self._min_max_values,
            'features': self._field_names,
            'result_names': self._result_names
        }
        with open(os.path.join(directory, f'normalized.json'), 'w', encoding='utf-8') as step_file:
            json.dump(config, step_file)
