import os
import sys
import math
import json
import logging

import tensorflow as tf
import numpy as np

from .knn_util import max_labels

logger = logging.getLogger(__name__)


class TensorflowKNNModel:
    """
    A KNN Implementation using numpy and matrix operations to achieve euclidean distance
    """
    def __init__(self, k=5, score_func='max_labels'):
        """
        :param distance_algorithm: Either euclidean or manhattan
        :param weight_function: The weight function, takes a distance and returns an adjusted distance
            that can be based on weight
        :param score_func: Options are max_labels...
        """
        self._k = k
        self._label_selection = max_labels if score_func == 'max_labels' else None
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

        self._feature_count = self._train_vectors.shape[1]

        self._define_kNearestNeighbors_model()

    def predict(self, vector):
        neighbors = self._kNearestNeighbors(vector)

        labels = [self._train_labels[n[0]] for n in neighbors]
        updated_weights = [n[1] for n in neighbors]

        return self._label_selection([(n[0], l) for n, l in zip(updated_weights, labels)])

    def _define_kNearestNeighbors_model(self):
        input_shape = (1, self._feature_count)
        output_shape = (len(self._train_vectors),)

        knn_graph = tf.Graph()
        with knn_graph.as_default() as g:
            X = tf.placeholder(tf.float32, input_shape)   # Shape(1, feature_count)

            neighbor_constant = tf.constant(self._train_vectors)  # Shape(N, feature_count)

            diff_op = X - neighbor_constant   # Shape(N, feature_count)
            squares = tf.square(diff_op)      # Shape(N, feature_count)
            sums = tf.matmul(squares, tf.ones((self._feature_count, 1)))  # Shape(N, 1)
            roots = tf.sqrt(sums)   # Shape(N, 1)
            model = tf.reshape(roots, output_shape)  # Shape(N,)

            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)

        self._model = model
        self._X = X
        self._sess = sess

    def _kNearestNeighbors(self, point):
        point_vector = np.array(point, dtype=np.float32).reshape(1, 4)   # Shape(1, feature_count)
        distance_scores = self._sess.run(self._model, feed_dict={self._X: point_vector})
        distances = list(enumerate(zip(distance_scores, self._train_vectors)))
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
