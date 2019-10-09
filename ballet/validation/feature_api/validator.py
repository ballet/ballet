from ballet.contrib import _collect_contrib_features
from ballet.util.log import logger
from ballet.validation.base import BaseValidator
from ballet.validation.common import (
    ChangeCollector, check_from_class, subsample_data_for_validation)
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


class FeatureApiValidator(BaseValidator):

    def __init__(self, project):
        self.change_collector = ChangeCollector(project)

        X, y = project.load_data()
        self.X, self.y = subsample_data_for_validation(X, y)

    def validate(self):
        """Collect and validate all new features"""

        changes = self.change_collector.collect_changes()

        features = []
        imported_okay = True
        for importer, modname, modpath in changes.new_feature_info:
            try:
                mod = importer()
                features.extend(_collect_contrib_features(mod))
            except (ImportError, SyntaxError):
                logger.info(
                    'Failed to import module at {}'
                    .format(modpath))
                logger.exception('Exception details: ')
                imported_okay = False

        if not imported_okay:
            return False

        # if no features were added at all, reject
        if not features:
            logger.info('Failed to collect any new features.')
            return False

        return all(
            validate_feature_api(feature, self.X, self.y, subsample=False)
            for feature in features
        )
