import os

import numpy as np


class KFoldTrainer:
    def __init__(self, model=None, data=None, evaluator=None, k=5):
        self._k = k
        self._model = model
        self._data = data
        self._evaluator = evaluator

    def _chunks(self, data):
        data_length = len(data)
        split_size = int(data_length / self._k)

        for i in range(0, data_length, split_size):
            yield data[i:i + split_size]

    def __call__(self):
        train_data, _  = self._data(split=1)
        train_data = list(train_data)
        train_splits = list(self._chunks(train_data))

        for current_k in range(self._k):
            self._run_split(current_k, train_splits)

    def _run_split(self, split_num, train_splits):
        validation = train_splits[split_num]
        train = [t for i, ts in enumerate(train_splits) for t in ts if i != split_num]
        self._model.fit(train)
        self._evaluator(self._model, validation)

    def store_results(self, directory):
        os.makedirs(directory, exist_ok=True)
        self._data.store_results(os.path.join(directory, 'data_output'))
        self._model.store_results(os.path.join(directory, 'model_output'))
        self._evaluator.store_results(os.path.join(directory, 'evaluator_output'))
