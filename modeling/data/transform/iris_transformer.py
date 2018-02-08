import os
import json


class IrisTransformer:
    def __init__(self, reader, features=['petal_width', 'petal_length', 'sepal_length', 'sepal_width'], label_name='species'):
        self._reader = reader
        self._features = features
        self._label_name = label_name

    def __call__(self, split=0.2):
        train_data, validate_data = self._reader(split)

        return self._transform_gen(train_data), self._transform_gen(validate_data)

    def _transform_gen(self, data_sequence):
        for d in data_sequence:
            yield ([d[f] for f in self._features], d[self._label_name])

    def store_results(self, directory):
        os.makedirs(directory, exist_ok=True)
        self._reader.store_results(directory)

        transform_config = {
            'features': self._features,
            'label_name': self._label_name,
        }
        with open(os.path.join(directory, 'transform.json'), 'w', encoding='utf-8') as transform_file:
            json.dump(transform_config, transform_file, indent=4, ensure_ascii=False)
