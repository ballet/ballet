from ballet import logger
from ballet.validation.common import subsample_data_for_validation, \
    check_from_class
from ballet.validation.feature_api.checks import FeatureApiCheck


def validate_feature_api(feature, X, y, subsample=False):
    logger.debug('Validating feature {feature!r}'.format(feature=feature))
    if subsample:
        X, y = subsample_data_for_validation(X, y)
    valid, failures = check_from_class(FeatureApiCheck, feature, X, y)
    if valid:
        logger.info('Feature is valid')
    else:
        logger.info(
            'Feature is NOT valid; failures were {failures}'
            .format(failures=failures))
    return valid
