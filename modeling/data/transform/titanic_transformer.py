import os
import json


class TitanicTransformer:
    """
    The data from the titanic reader has a number of features.

    Numeric:
    - PassengerId
    - Pclass
    - Age
    - SibSp
    - Parch
    - Ticket
    - Fare

    Non-Numeric:
    - Name
    - Sex
    - Cabin
    - Embarked
    """
    def __init__(self, reader,
                 num_features=['Age', 'Fare'],
                 category_features=['Sex', 'Embarked', 'Pclass', 'SibSp', 'Parch'],
                 use_has_cabin=False,
                 label_name='Survived'):
        self._reader = reader
        self._label_name = label_name
        self._num_features = num_features
        self._category_features = category_features
        self._use_has_cabin = use_has_cabin

        self._category_values = {
            'Sex': ('male', 'female'),
            'Embarked': ('', 'Q', 'C', 'S'),
            'Pclass': ('1', '2', '3'),
            'SibSp': ('0', '1', '2', '3', '4', '5', '8'),
            'Parch': ('0', '1', '2', '3', '4', '5', '6', '9')
        }

    def __call__(self, split=0.2):
        train_data, validate_data = self._reader(split)

        return self._transform_gen(train_data), self._transform_gen(validate_data)

    def _transform_gen(self, data_sequence):
        for d in data_sequence:
            vector = []
            for n in self._num_features:
                vector.append(float(d[n]) if d[n] else -1)

            if self._use_has_cabin:
                vector.append(int(d['Cabin'] != ''))

            for c in self._category_features:
                vector.append(self._category_values[c].index(d[c]))

            yield vector, d[self._label_name]

    def store_results(self, directory):
        os.makedirs(directory, exist_ok=True)
        self._reader.store_results(directory)

        transform_config = {
            'num_features': self._num_features,
            'category_features': self._category_features,
            'label_name': self._label_name,
            'use_has_cabin': self._use_has_cabin,
            'category_values': self._category_values,
        }
        with open(os.path.join(directory, 'transform.json'), 'w', encoding='utf-8') as transform_file:
            json.dump(transform_config, transform_file, indent=4, ensure_ascii=False)

