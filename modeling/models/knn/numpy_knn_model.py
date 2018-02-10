import os
import sys
import math
import json
import logging
import heapq

from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)


class NumpyKNNModel:
    """
    A KNN Implementation using numpy and matrix operations to achieve euclidean distance
    """
    def max_labels(distance_labels):
        label_counter = Counter([d[1] for d in distance_labels])
        most_common = label_counter.most_common()[0]
        label_counts = len(distance_labels)
        return (most_common[0], most_common[1] / label_counts)

    def __init__(self, k=5, score_func='max_labels'):
        """
        :param distance_algorithm: Either euclidean or manhattan
        :param weight_function: The weight function, takes a distance and returns an adjusted distance
            that can be based on weight
        :param score_func: Options are max_labels...
        """
        self._k = k
        self._label_selection = NumpyKNNModel.max_labels if score_func else None
        self._train_vectors = None
        self._train_labels = None
        self._config = {
            'k': k,
            'score_func': score_func
        }

    def fit(self, train_data):
        """
        :param train_vectors: Sequence of Vectors[float] to be used in training
        :param train_labels: Sequence of label values (any type)
        """
        train_data = list(train_data)
        train_vectors = np.array([t[0] for t in train_data], dtype=np.float32)
        train_labels = np.array([t[1] for t in train_data])

        self._train_vectors = train_vectors
        self._train_labels = train_labels

    def predict(self, vector):
        neighbors = self._kNearestNeighbors(vector)

        labels = [self._train_labels[n[0]] for n in neighbors]
        updated_weights = [n[1] for n in neighbors]

        return self._label_selection([(n[0], l) for n, l in zip(updated_weights, labels)])

    def _kNearestNeighbors(self, point):
        point_vector = np.array(point, dtype=np.float32)          # Shape(1, feature_count)
        difference = point_vector - self._train_vectors           # Shape(N, feature_count)
        squares = np.square(difference)                           # Shape(N, feature_count): squares = difference * difference
        sums = squares @ np.ones(shape=point_vector.T.shape)      # Shape(feature_count, 1): np.sum(squares, axis=1) or squares.dot(np.ones(shape=point_vector.T.shape))
        # @/dot seems to be faster then np.sum - https://stackoverflow.com/questions/37356645/numpy-is-matrix-multiplication-faster-than-sum-of-a-vector
        roots = np.sqrt(sums)
        distances = list(enumerate(zip(roots, self._train_vectors)))
        sorted_nearest = sorted(distances, key=lambda x: x[1][0])

        if self._k:
            return sorted_nearest[:self._k]
        else:
            return sorted_nearest

    def store_results(self, directory):
        os.makedirs(directory, exist_ok=True)

        with open(os.path.join(directory, 'config.json'), 'w', encoding='utf-8') as config_file:
            json.dump(self._config, config_file, indent=4, ensure_ascii=False)

        with open(os.path.join(directory, 'train_data.json'), 'w', encoding='utf-8') as train_file:
            json.dump(list(zip(self._train_vectors.tolist(), self._train_labels.tolist())), train_file, indent=4, ensure_ascii=False)
