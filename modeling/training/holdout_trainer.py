import os

import numpy as np


class HoldoutTrainer:
    def __init__(self, model=None, data=None, evaluator=None, holdout=0.2):
        self._holdout = holdout
        self._model = model
        self._data = data
        self._evaluator = evaluator

    def __call__(self):
        train_data, validate_data = self._data(split=(1-self._holdout))

        self._model.fit(train_data)

        self._evaluator(self._model, validate_data)

    def store_results(self, directory):
        os.makedirs(directory, exist_ok=True)
        self._data.store_results(os.path.join(directory, 'data_output'))
        self._model.store_results(os.path.join(directory, 'model_output'))
        self._evaluator.store_results(os.path.join(directory, 'evaluator_output'))
