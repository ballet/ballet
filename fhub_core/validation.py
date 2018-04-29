import logging
import pathlib

import funcy

from fhub_core.contrib import get_contrib_features
from fhub_core.feature import FeatureValidator
from fhub_core.util.git import (
    HeadInfo, PullRequestInfo, get_file_changes_by_revision, )
from fhub_core.util.modutil import import_module_from_relpath

logger = logging.getLogger(__name__)

__all__ = [
    'PullRequestFeatureValidator'
]


class PullRequestFeatureValidator:
    def __init__(self, pr_num, repo, comparison_ref, contrib_module_path,
                 X_df, y_df):
        '''Validate the features introduced in a proposed pull request

        Args:
            pr_num (str): Pull request number
            repo (git.Repo): Project repository
            comparison_ref (str): Name of comparison ref, e.g. 'master'
            contrib_module_path (str): Relative path to contrib module
            X_df (pd.DataFrame): Example X DataFrame
            y_df (pd.DataFrame): Example y DataFrame
        '''
        self.pr_num = pr_num
        self.repo = repo
        self.comparison_ref = comparison_ref
        self.contrib_module_path = contrib_module_path
        self.X_df = X_df
        self.y_df = y_df

        self.pr_info = PullRequestInfo(self.pr_num)
        self.head_info = HeadInfo(self.repo)

        # may be set by other methods
        self.file_changes = None
        self.file_changes_admissible = None
        self.file_changes_inadmissible = None
        self.features = None

    def collect_file_changes(self):
        logger.info('Collecting file changes...')

        from_rev = self.comparison_ref
        to_rev = self.pr_info.local_rev_name
        self.file_changes = get_file_changes_by_revision(
            self.repo, from_rev, to_rev)

        # log results
        for i, file in enumerate(self.file_changes):
            logger.debug('File {i}: {file}'.format(i=i, file=file))
        logger.info('Collected {} files'.format(len(self.file_changes)))

    def categorize_file_changes(self):
        '''Partition file changes into admissible and inadmissible changes'''
        if self.file_changes is None:
            raise ValueError('File changes have not been collected.')

        logger.info('Categorizing file changes...')

        self.file_changes_admissible = []
        self.file_changes_inadmissible = []

        # admissible:
        # - within contrib subdirectory
        # - is a .py file
        # - TODO: is a .txt file
        # - is an addition
        # inadmissible:
        # - otherwise (wrong directory, wrong filetype, wrong modification
        #   type)
        def within_contrib_subdirectory(file):
            contrib_relpath = self.contrib_module_path
            return pathlib.Path(contrib_relpath) in pathlib.Path(file).parents

        def is_appropriate_filetype(file):
            return file.endswith('.py')

        def is_appropriate_modification_type(modification_type):
            # TODO
            # return modification_type == 'A'
            return True

        is_admissible = funcy.all_fn(
            within_contrib_subdirectory, is_appropriate_filetype,
            is_appropriate_modification_type)

        for file in self.file_changes:
            if is_admissible(file):
                self.file_changes_admissible.append(file)
                logger.debug(
                    'Categorized {file} as ADMISSIBLE'.format(file=file))
            else:
                self.file_changes_inadmissible.append(file)
                logger.debug(
                    'Categorized {file} as INADMISSIBLE'.format(file=file))

        logger.info('Admitted {} files and rejected {} files'.format(
            len(self.file_changes_admissible),
            len(self.file_changes_inadmissible)))

    def collect_features(self):
        if self.file_changes_admissible is None:
            raise ValueError('File changes have not been collected.')

        logger.info('Collecting features...')

        self.features = []
        for file in self.file_changes_admissible:
            try:
                mod = import_module_from_relpath(file)
            except ImportError:
                logger.exception(
                    'Failed to import module from {}'.format(file))
                continue

            features = get_contrib_features(mod)
            self.features.extend(features)

        logger.info('Collected {} features'.format(len(self.features)))

    def validate_features(self, features):
        # get small subset?
        X_df, y_df = subsample_data_for_validation(self.X_df, self.y_df)

        # validate
        feature_validator = FeatureValidator(X_df, y_df)
        overall_result = True
        for feature in features:
            result, failures = feature_validator.validate(feature)
            if result is True:
                logger.info(
                    'Feature is valid: {feature}'.format(feature=feature))
            else:
                logger.info(
                    'Feature is NOT valid: {feature}'.format(feature=feature))
                logger.debug(
                    'Failures in validation: {failures}'
                    .format(failures=failures))
                overall_result = False

        return overall_result

    def validate(self):
        # check that we are *on* this PR's branch
        expected_ref = self.pr_info.local_rev_name
        current_ref = self.head_info.path
        if expected_ref != current_ref:
            raise NotImplementedError(
                'Must validate PR while on that PR\'s branch')

        # collect
        self.collect_file_changes()
        self.categorize_file_changes()
        self.collect_features()

        # validate
        result = self.validate_features(self.features)

        return result


def subsample_data_for_validation(X_df_tr, y_df_tr):
    # TODO
    return X_df_tr, y_df_tr
