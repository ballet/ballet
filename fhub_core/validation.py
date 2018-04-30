import copy
import logging
import pathlib

import funcy

from fhub_core.contrib import get_contrib_features
from fhub_core.feature import Feature
from fhub_core.util import assertion_method
from fhub_core.util.gitutil import get_file_changes_by_revision
from fhub_core.util.modutil import import_module_from_relpath

logger = logging.getLogger(__name__)

__all__ = [
    'FeatureValidator',
    'PullRequestFeatureValidator'
]


class FeatureValidator:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    @assertion_method
    def _is_feature(self, feature):
        assert isinstance(feature, Feature)

    @assertion_method
    def _has_correct_input_type(self, feature):
        '''Check that `input` is a string or iterable of string'''
        input = feature.input
        is_str = funcy.isa(str)
        is_nested_str = funcy.all_fn(
            funcy.iterable, lambda x: all(map(is_str, x)))
        assert is_str(input) or is_nested_str(input)

    @assertion_method
    def _has_transformer_interface(self, feature):
        assert hasattr(feature.transformer, 'fit')
        assert hasattr(feature.transformer, 'transform')

    @assertion_method
    def _can_make_mapper(self, feature):
        try:
            feature.as_dataframe_mapper()
        except Exception:
            raise AssertionError

    @assertion_method
    def _can_fit(self, feature):
        try:
            mapper = feature.as_dataframe_mapper()
            mapper.fit(self.X, self.y)
        except Exception:
            raise AssertionError

    @assertion_method
    def _can_transform(self, feature):
        try:
            mapper = feature.as_dataframe_mapper()
            mapper.fit(self.X, self.y)
            mapper.transform(self.X)
        except Exception:
            raise AssertionError

    @assertion_method
    def _can_fit_transform(self, feature):
        try:
            mapper = feature.as_dataframe_mapper()
            mapper.fit_transform(self.X, self.y)
        except Exception:
            raise AssertionError

    @assertion_method
    def _has_correct_output_dimensions(self, feature):
        try:
            mapper = feature.as_dataframe_mapper()
            X = mapper.fit_transform(self.X, self.y)
        except Exception:
            raise AssertionError

        assert self.X.shape[0] == X.shape[0]

    @assertion_method
    def _can_deepcopy(self, feature):
        try:
            copy.deepcopy(feature)
        except Exception:
            raise AssertionError

    def _get_all_checks(self):
        for method_name in self.__dir__():
            method = getattr(self, method_name)
            if hasattr(method, 'is_check') and method.is_check:
                name = method.__name__
                if name.startswith('_'):
                    name = name[1:]
                yield (method, name)

    def validate(self, feature):
        failures = []
        result = True
        for check, name in self._get_all_checks():
            success = check(feature)
            if not success:
                result = False
                failures.append(name)

        return result, failures


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

        # self.pr_info = PullRequestInfo(self.pr_num)
        # self.head_info = HeadInfo(self.repo)
        self.pr_head = 'HEAD'

        # will be set by other methods
        self.file_changes = None
        self.file_changes_admissible = None
        self.file_changes_inadmissible = None
        self.file_changes_validation_result = None
        self.features = None
        self.features_validation_result = None

    def _collect_file_changes(self):
        logger.info('Collecting file changes...')

        from_rev = self.comparison_ref
        to_rev = self.pr_head
        self.file_changes = get_file_changes_by_revision(
            self.repo, from_rev, to_rev)

        # log results
        for i, file in enumerate(self.file_changes):
            logger.debug('File {i}: {file}'.format(i=i, file=file))
        logger.info('Collected {} file(s)'.format(len(self.file_changes)))

    def _categorize_file_changes(self):
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

    def _validate_files(self):
        if self.file_changes_inadmissible is None:
            raise ValueError('File changes have not been categorized.')

        result = len(self.file_changes_inadmissible) == 0
        self.file_changes_validation_result = result

    def _collect_features(self):
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

        logger.info('Collected {} feature(s)'.format(len(self.features)))

    def _validate_features(self):
        if self.features is None:
            raise ValueError('Features have not been collected.')

        # get small subset?
        X_df, y_df = subsample_data_for_validation(self.X_df, self.y_df)

        # validate
        feature_validator = FeatureValidator(X_df, y_df)
        overall_result = True
        for feature in self.features:
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

        self.features_validation_result = overall_result

    def _determine_validation_result(self):
        if self.file_changes_validation_result is None:
            raise ValueError('File changes have not been validated.')
        if self.features_validation_result is None:
            raise ValueError('Feature changes have not been validated.')
        return (self.file_changes_validation_result and
                self.features_validation_result)

    def validate(self):
        # # check that we are *on* this PR's branch
        # expected_ref = self.pr_info.local_rev_name
        # current_ref = self.head_info.path
        # if expected_ref != current_ref:
        #     raise NotImplementedError(
        #         'Must validate PR while on that PR\'s branch')

        # collect, categorize, and validate file changes
        self._collect_file_changes()
        self._categorize_file_changes()
        self._validate_files()

        # collect and validate new features
        self._collect_features()
        self._validate_features()

        # determine overall result
        overall_result = self._determine_validation_result()
        return overall_result


def subsample_data_for_validation(X_df_tr, y_df_tr):
    # TODO
    return X_df_tr, y_df_tr
