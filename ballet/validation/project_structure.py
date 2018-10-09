import ballet
from ballet.util.log import logger
from ballet.validation.base import ProjectStructureValidator


class FileChangeValidator(ProjectStructureValidator):

    def validate(self):
        _, _, inadmissible, _, imported_okay = self.collect_changes()
        return not inadmissible and imported_okay


def subsample_data_for_validation(X, y):
    return X, y


class FeatureApiValidator(ProjectStructureValidator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.X, self.y = subsample_data_for_validation(self.X, self.y)

    def validate(self):
        """Collect and validate all new features"""

        _, _, _, new_features, _ = self.collect_changes()

        # if no features were added at all, reject
        if not new_features:
            logger.info('Failed to collect any new features.')
            return False

        # validate
        okay = True
        for feature in new_features:
            result, failures = ballet.validation.feature_api.validate(
                feature, self.X, self.y)
            if result is True:
                logger.info(
                    'Feature is valid: {feature}'.format(feature=feature))
            else:
                logger.info(
                    'Feature is NOT valid: {feature}'.format(feature=feature))
                logger.debug(
                    'Failures in validation: {failures}'
                    .format(failures=failures))
                okay = False

        return okay
