import os

from typing import Sequence


class Transformer:
    def __init__(self, model, steps):
        self._model = model
        self._steps = steps

    def fit(self, train_data):
        if self._two_pass():
            train_data = list(train_data)
            for s in self._steps:
                if hasattr(s, 'prep_pass'):
                    s.prep_pass(train_data)

        self._model.fit(self._transform_generator(train_data))

    def predict(self, point):
        return self._model.predict(self._transform(point))

    def _two_pass(self):
        return any([s.two_pass() for s in self._steps if hasattr(s, 'two_pass')])

    def _transform_generator(self, train_data):
        for p, l in train_data:
            yield (self._transform(p), l)

    def _transform(self, point):
        meta = {}
        for s in self._steps:
            meta = s(point, meta)

        vector = []
        for _, v in meta.items():
            if isinstance(v, Sequence):
                vector.extend(v)
            else:
                vector.append(v)

        return vector

    def store_results(self, directory):
        steps_directory = os.path.join(directory, 'steps')
        os.makedirs(steps_directory, exist_ok=True)

        for s in self._steps:
            s.store_results(steps_directory)

        self._model.store_results(os.path.join(directory, 'model'))
