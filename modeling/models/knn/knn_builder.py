import logging

from .knn_model import ProceduralKNNModel
from .numpy_knn_model import NumpyKNNModel
from .tf_knn_model import TensorflowKNNModel

logger = logging.getLogger(__name__)


def knn_model(k=5, implementation='tf', **kwargs):
    if implementation == 'tf':
        logger.debug(f'Creating implementation of TensorflowKNN - {k} - {kwargs}')
        model = TensorflowKNNModel(k=k, **kwargs)
    elif implementation == 'numpy':
        logger.debug(f'Creating implementation of NumpyKNN - {k} - {kwargs}')
        model = NumpyKNNModel(k=k, **kwargs)
    else:
        logger.debug(f'Creating implementation of ProceduralKNN - {k} - {kwargs}')
        model = ProceduralKNNModel(k=k, **kwargs)

    return model
