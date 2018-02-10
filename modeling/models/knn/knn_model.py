import os
import math
import json
import logging

from collections import Counter

logger = logging.getLogger(__name__)


class KNNModel:
    def euclidean(point_one, point_two):
        square_sum = sum([(p-q) ** 2 for p, q in zip(point_one, point_two)])
        return math.sqrt(square_sum)

    def manhattan(point_one, point_two):
        return sum([abs(p-q) for p, q in zip(point_one, point_two)])

    def max_labels(distance_labels):
        label_counter = Counter([d[1] for d in distance_labels])
        most_common = label_counter.most_common()[0]
        label_counts = len(distance_labels)
        return (most_common[0], most_common[1] / label_counts)

    def __init__(self, k=5, distance_algorithm='euclidean', score_func='max_labels'):
        """
        :param k: The number of neighbors to include
        :param distance_algorithm: Either euclidean or manhattan
        :param score_func: Options are max_labels...
        """
        self._k = k
        self._distance_algorithm = KNNModel.manhattan if 'manhattan' else KNNModel.euclidean
        self._label_selection = KNNModel.max_labels if score_func else None
        self._train_vectors = None
        self._train_labels = None
        self._config = {
            'k': k,
            'distance_algorithm': distance_algorithm,
            'score_func': score_func
        }

    def fit(self, train_data):
        """
        :param train_vectors: Sequence of Vectors[float] to be used in training
        :param train_labels: Sequence of label values (any type)
        """
        train_data = list(train_data)
        train_vectors = [t[0] for t in train_data]
        train_labels = [t[1] for t in train_data]

        self._train_vectors = train_vectors
        self._train_labels = train_labels

    def predict(self, vector):
        neighbors = self._kNearestNeighbors(vector)

        labels = [self._train_labels[n[0]] for n in neighbors]
        updated_weights = [n[1] for n in neighbors]

        return self._label_selection([(n[0], l) for n, l in zip(updated_weights, labels)])

    def _kNearestNeighbors(self, point):
        nearest = []
        for i, data_point in enumerate(self._train_vectors):
            distance = self._distance_algorithm(point, data_point)
            nearest.append((i, (distance, data_point)))

        sorted_nearest = sorted(nearest, key=lambda x: x[1][0])
        if self._k:
            return sorted_nearest[:self._k]
        else:
            return sorted_nearest

    def store_results(self, directory):
        os.makedirs(directory, exist_ok=True)

        with open(os.path.join(directory, 'config.json'), 'w', encoding='utf-8') as config_file:
            json.dump(self._config, config_file, indent=4, ensure_ascii=False)

        with open(os.path.join(directory, 'train_data.json'), 'w', encoding='utf-8') as train_file:
            json.dump(list(zip(self._train_vectors, self._train_labels)), train_file, indent=4, ensure_ascii=False)
