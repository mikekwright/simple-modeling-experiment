import os
import logging
import csv
import json

import numpy as np

logger = logging.getLogger(__name__)


class TitanicReader:
    TEST_NAME = 'test.csv'
    TRAIN_NAME = 'train.csv'

    def __init__(self, directory, include_test=False, seed=42):
        self._directory = directory
        self._include_test = include_test
        self._seed = seed
        self._random = np.random.RandomState(seed=seed)

    def _read_file(self, filename):
        with open(os.path.join(self._directory, filename), 'r') as current_file:
            contents = [dict(r) for r in csv.DictReader(current_file)]
        return contents

    def __call__(self, split=0.2):
        all_data = self._read_file(TitanicReader.TRAIN_NAME)

        if self._include_test:
            test_data = self._read_file(TitanicReader.TEST_NAME)
            all_data = all_data + test_data

        self._random.shuffle(all_data)
        split_index = int(len(all_data) * split)

        return all_data[:split_index], all_data[split_index:]

    def store_results(self, directory):
        os.makedirs(directory, exist_ok=True)

        data_config = {
            'seed': self._seed,
            'directory': self._directory
        }
        with open(os.path.join(directory, 'config.json'), 'w', encoding='utf-8') as config_file:
            json.dump(data_config, config_file, indent=4, ensure_ascii=False)
