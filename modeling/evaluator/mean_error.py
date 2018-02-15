import logging
import os
import json

import numpy as np

logger = logging.getLogger(__name__)


class MeanErrorEvaluator:
    def __init__(self, me_type='MSE'):
        """
        :param me_type: The Mean Error type to use, options are (MSE, RMSE, MAE)
        """
        self._me_type = me_type
        self._prediction_results = []
        self._errors = []
        self._run_count = 0

    def __call__(self, model, validation_data):
        self._run_count += 1

        prediction_details = []
        all_predictions = []
        all_actuals = []
        for i, v in enumerate(validation_data):
            prediction = model.predict(v[0])
            actual = v[1]

            single_error = self._me_result([prediction[0]], [actual])
            result_object = self._create_result_object(i, v[0], prediction[0], v[1], single_error)
            prediction_details.append(result_object)

            all_predictions.append(prediction[0])
            all_actuals.append(actual)

        error = self._me_result(all_predictions, all_actuals)
        logger.info(f'Run {self._run_count} - {self._me_type}: {error}')
        self._prediction_results.append(prediction_details)
        self._errors.append(error)

    def _me_result(self, predictions, actuals):
        operation = abs if self._me_type == 'MAE' else lambda x: np.power(x, 2)

        n = len(predictions)
        errors = [operation(y-yh) for yh, y in zip(predictions, actuals)]
        result = np.sum(errors) / n
        if self._me_type == 'RMSE':
            return float(np.sqrt(result))
        return float(result)

    def _create_result_object(self, index, vector, prediction, expected, error):
        return {
            'index': index,
            'vector': vector,
            'prediction': prediction,
            'expected': expected,
            'error': error,
            'run': self._run_count,
        }

    def store_results(self, directory):
        os.makedirs(directory, exist_ok=True)

        score_details = {
            'error': self._errors if len(self._errors) > 1 else self._errors[0],
        }
        with open(os.path.join(directory, 'score.json'), 'w', encoding='utf-8') as score_file:
            json.dump(score_details, score_file, indent=4, ensure_ascii=False)

        with open(os.path.join(directory, 'predictions.json'), 'w', encoding='utf-8') as predictions_file:
            json.dump(self._prediction_results, predictions_file, indent=4, ensure_ascii=False)
