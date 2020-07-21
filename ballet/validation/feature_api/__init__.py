import pandas as pd

from ballet import logger
from ballet.feature import Feature
from ballet.validation.common import (
    check_from_class, subsample_data_for_validation)
from ballet.validation.feature_api.checks import FeatureApiCheck


def validate_feature_api(
    feature: Feature,
    X_df: pd.DataFrame,
    y: pd.DataFrame,
    subsample: bool,
) -> bool:
    logger.debug('Validating feature {feature!r}'.format(feature=feature))
    if subsample:
        X_df, y = subsample_data_for_validation(X_df, y)
    valid, failures = check_from_class(FeatureApiCheck, feature, X_df, y)
    if valid:
        logger.info('Feature is valid')
    else:
        logger.info(
            'Feature is NOT valid; failures were {failures}'
            .format(failures=failures))
    return valid
