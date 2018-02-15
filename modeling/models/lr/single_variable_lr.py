import os
import json

import numpy as np

class SVLinearRegression:
    def __init__(self):
        self._slope = None
        self._intercept = None

    def fit(self, data):
        data = list(data)
        X = [d[0][0] for d in data]
        Y = [d[1] for d in data]

        x_mean = np.mean(X)
        y_mean = np.mean(Y)

        x_deviation = [x - x_mean for x in X]
        y_deviation = [y - y_mean for y in Y]
        sum_deviation_product = sum([x * y for x, y in zip(x_deviation, y_deviation)])
        sum_squared_x_deviation = sum([np.power(x, 2) for x in x_deviation])

        self._slope = b1 = sum_deviation_product / sum_squared_x_deviation
        self._intercept = y_mean - (b1 * x_mean)

    def predict(self, x):
        return (self._intercept + (self._slope * x[0]), 0.0)

    def store_results(self, directory):
        os.makedirs(directory, exist_ok=True)

        config = {
            'slope': float(self._slope),
            'intercept': float(self._intercept),
        }
        with open(os.path.join(directory, 'config.json'), 'w', encoding='utf-8') as config_file:
            json.dump(config, config_file, indent=4, ensure_ascii=False)
