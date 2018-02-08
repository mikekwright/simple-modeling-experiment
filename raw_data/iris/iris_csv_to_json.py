import os
import json
import csv

directory = os.path.dirname(__file__)
with open(os.path.join(directory, 'iris_data.csv'), 'r') as iris_file:
    iris_reader = csv.reader(iris_file)
    all_data = list(iris_reader)

iris_data = [{
    'sepal_length': float(d[0]),
    'sepal_width': float(d[1]),
    'petal_length': float(d[2]),
    'petal_width': float(d[3]),
    'species': d[4]
    }
    for d in all_data
]

with open(os.path.join(directory, 'iris_data.json'), 'w') as iris_file:
    json.dump(iris_data, iris_file, indent=4, ensure_ascii=False)