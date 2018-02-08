import os
import json
import logging

logger = logging.getLogger(__name__)


class AccuracyEvaluator:
    def __init__(self):
        self._correct_labels = []
        self._incorrect_labels = []
        self._accuracy = []
        self._run_count = 0

    def _create_result_object(self, index, vector, prediction, expected, confidence):
        return {
            'index': index,
            'vector': vector,
            'prediction': prediction,
            'expected': expected,
            'confidence': confidence,
            'run': self._run_count,
        }

    def __call__(self, model, validation_data):
        self._run_count += 1
        current_correct = []
        current_incorrect = []
        for i, v in enumerate(validation_data):
            prediction = model.predict(v[0])
            result_object = self._create_result_object(i, v[0], prediction[0], v[1], prediction[1])
            if prediction[0] == v[1]:
                current_correct.append(result_object)
            else:
                current_incorrect.append(result_object)

        current_accuracy = len(current_correct) / (len(current_incorrect) + len(current_correct))
        logger.info(f'Run {self._run_count} - Accuracy: {current_accuracy} - correct {len(current_correct)} incorrect {len(current_incorrect)}')

        self._correct_labels.append(current_correct)
        self._incorrect_labels.append(current_incorrect)
        self._accuracy.append(current_accuracy)

    def store_results(self, directory):
        os.makedirs(directory, exist_ok=True)

        score_details = {
            'accuracy': self._accuracy if len(self._accuracy) > 1 else self._accuracy[0],
        }
        with open(os.path.join(directory, 'score.json'), 'w', encoding='utf-8') as score_file:
            json.dump(score_details, score_file, indent=4, ensure_ascii=False)

        with open(os.path.join(directory, 'correct_labels.json'), 'w', encoding='utf-8') as correct_file:
            json_correct = self._correct_labels if len(self._correct_labels) > 1 else self._correct_labels[0]
            json.dump(json_correct, correct_file, indent=4, ensure_ascii=False)

        with open(os.path.join(directory, 'incorrect_labels.json'), 'w', encoding='utf-8') as incorrect_file:
            json_incorrect = self._incorrect_labels if len(self._incorrect_labels) > 1 else self._incorrect_labels[0]
            json.dump(json_incorrect, incorrect_file, indent=4, ensure_ascii=False)
