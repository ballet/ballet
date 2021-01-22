from typing import Union

import pandas as pd

from ballet import logger
from ballet.feature import Feature
from ballet.validation.common import (
    check_from_class, subsample_data_for_validation,)
from ballet.validation.feature_api.checks import FeatureApiCheck


def validate_feature_api(
    feature: Feature,
    X_df: pd.DataFrame,
    y_df: Union[pd.DataFrame, pd.Series],
    subsample: bool,
    log_advice: bool = False,
) -> bool:
    logger.debug(f'Validating feature {feature!r}')
    if subsample:
        X_df, y_df = subsample_data_for_validation(X_df, y_df)
    valid, failures, advice = check_from_class(
        FeatureApiCheck, feature, X_df, y_df)
    if valid:
        logger.info('Feature is valid')
    else:
        if log_advice:
            logger.info(
                'Feature is NOT valid; here is some advice for resolving the '
                'feature API issues.')
            for failure, advice_item in zip(failures, advice):
                logger.info(f'{failure}: {advice_item}')
        else:
            logger.info(f'Feature is NOT valid; failures were {failures}')

    return valid
