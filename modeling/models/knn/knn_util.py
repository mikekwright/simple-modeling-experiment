import math
import numpy as np

from collections import Counter

def max_labels(distance_labels):
    label_counter = Counter([d[1] for d in distance_labels])
    most_common = label_counter.most_common()[0]
    label_counts = len(distance_labels)
    return (most_common[0], most_common[1] / label_counts)

def euclidean(point_one, point_two):
    square_sum = sum([(p-q) ** 2 for p, q in zip(point_one, point_two)])
    return math.sqrt(square_sum)

def manhattan(point_one, point_two):
    return sum([abs(p-q) for p, q in zip(point_one, point_two)])
