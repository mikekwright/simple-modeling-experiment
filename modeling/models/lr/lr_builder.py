import logging

from .single_variable_lr import SVLinearRegression

logger = logging.getLogger(__name__)


def lr_model(model='sv_lr'):
    logger.debug(f'Creating implementation of SVLinearRegression')
    return SVLinearRegression()
