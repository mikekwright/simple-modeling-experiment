import os
import json

import numpy as np
from typing import Dict

from .base import BaseStep


class FieldIsNotEmpty(BaseStep):
    def __init__(self, field_names, result_names=None):
        super().__init__(field_names, result_names)

    def _handle_call(self, field, value):
        return True if value else False

    def store_results(self, directory):
        pass


class ReplaceMissing(BaseStep):
    def __init__(self, field_names, result_names=None, operation='avg'):
        """
        Valid options for operation are (avg, min, max)
        """
        super().__init__(field_names, result_names)
        self._operation = operation
        self._replace_values = {}

    def two_pass(self):
        return True

    def prep_pass(self, train_data):
        for field in self.field_names:
            values = [float(t[0][field]) for t in train_data if t[0][field]]
            if self._operation == 'min':
                replace_value = min(values)
            elif self._operation == 'max':
                replace_value = max(values)
            else:
                replace_value = float(np.average(values))

            self._replace_values[field] = replace_value

    def _handle_call(self, field, value):
        if not value and value != 0:
            return self._replace_values[field]
        else:
            return value

    def store_results(self, directory):
        step_config = {
            'operation': self._operation,
            'replacements': self._replace_values
        }
        with open(os.path.join(directory, 'replace_missing.json'), 'w') as cf:
            json.dump(step_config, cf, indent=4)


class FillMissing(BaseStep):
    def __init__(self, field_names, replacements: Dict, result_names=None):
        """
        """
        if len(replacements) != len(field_names):
            raise ValueError('The replacement list is not the same size as the field_names')

        super().__init__(field_names, result_names)
        self._replacements = replacements

    def _handle_call(self, field, value):
        if not value and value != 0:
            return self._replacements[field]
        else:
            return value

    def store_results(self, directory):
        pass
