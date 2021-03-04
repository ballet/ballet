from ballet.contrib import _collect_contrib_features
from ballet.project import Project
from ballet.util.log import logger
from ballet.validation.base import BaseValidator
from ballet.validation.common import (
    ChangeCollector, subsample_data_for_validation,)
from ballet.validation.feature_api import validate_feature_api


class FeatureApiValidator(BaseValidator):

    def __init__(self, project: Project):
        self.change_collector = ChangeCollector(project)

        X_df, y_df = project.api.load_data()
        encoder = project.api.encoder
        y = encoder.fit_transform(y_df)
        self.X_df, self.y = subsample_data_for_validation(X_df, y)

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
                logger.info(f'Failed to import module at {modpath}')
                logger.exception('Exception details: ')
                imported_okay = False

        if not imported_okay:
            return False

        # if no features were added at all, reject
        if not features:
            logger.info('Failed to collect any new features.')
            return False

        return all(
            validate_feature_api(feature, self.X_df, self.y, False)
            for feature in features
        )
