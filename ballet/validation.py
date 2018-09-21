import copy
import pathlib

import funcy

from ballet.contrib import get_contrib_features
from ballet.exc import UnexpectedValidationStateError
from ballet.feature import Feature
from ballet.util import assertion_method
from ballet.util.gitutil import LocalPullRequestBuildDiffer
from ballet.util.log import logger
from ballet.util.modutil import import_module_at_path, relpath_to_modname
from ballet.util.travisutil import (
    TravisPullRequestBuildDiffer, can_use_travis_differ)

__all__ = [
    'FeatureValidator',
    'PullRequestFeatureValidator'
]


class FeatureValidator:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def validate(self, feature):
        '''Validate the feature'''
        failures = []
        result = True
        for check, name in self._get_all_checks():
            success = check(feature)
            if not success:
                result = False
                failures.append(name)

        return result, failures

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


class PullRequestFeatureValidator:
    APPROPRIATE_CHANGE_TYPES = ['A']
    APPROPRIATE_FILE_EXTS = ['.py']

    def __init__(self, repo, pr_num, contrib_module_path, X, y):
        '''Validate the features introduced in a proposed pull request.

        Args:
            repo (git.Repo): project repo
            pr_num (str): Pull request number
            contrib_module_path (str): Relative path to contrib module
            X (array-like): Example X array-like
            y (array-like): Example y array-like
        '''
        self.repo = repo
        self.pr_num = pr_num
        self.contrib_module_path = contrib_module_path
        self.X = X
        self.y = y

        if can_use_travis_differ():
            self.differ = TravisPullRequestBuildDiffer(self.pr_num)
        else:
            self.differ = LocalPullRequestBuildDiffer()

        # will be set by other methods
        self.file_diffs = None
        self.file_diffs_admissible = None
        self.file_diffs_inadmissible = None
        self.file_diffs_validation_result = None
        self.features = None
        self.features_validation_result = None

    def validate(self):
        '''Validate pull request.

        To do this, follows these steps:
        1. Collects the files that have changed in this pull request as
           compared to a comparison branch.
        2. Categorize these file changes into admissible or inadmissible file
           changes. Admissible file changes solely contribute python files to
           the contrib subdirectory.
        3. Collect features from admissible new files.
        4. Validate each of these features using the FeatureValidator.
        5. Report the overall validation results.
        '''

        # collect, categorize, and validate file changes
        self._collect_file_diffs()
        self._categorize_file_diffs()
        self._validate_files()

        # collect and validate new features
        self._collect_features()
        self._validate_features()

        # determine overall result
        overall_result = self._determine_validation_result()
        return overall_result

    def _collect_file_diffs(self):
        logger.info('Collecting file changes...')

        self.file_diffs = self.differ.diff()

        # log results
        for i, file in enumerate(self.file_diffs):
            logger.debug('File {i}: {file}'.format(i=i, file=file))
        logger.info('Collected {} file(s)'.format(len(self.file_diffs)))

    def _categorize_file_diffs(self):
        '''Partition file changes into admissible and inadmissible changes'''
        if self.file_diffs is None:
            raise UnexpectedValidationStateError(
                'File changes have not been collected.')

        logger.info('Categorizing file changes...')

        self.file_diffs_admissible = []
        self.file_diffs_inadmissible = []

        def is_appropriate_change_type(diff):
            '''File change is an addition'''
            return diff.change_type in \
                PullRequestFeatureValidator.APPROPRIATE_CHANGE_TYPES

        def within_contrib_subdirectory(diff):
            '''File addition is a subdirectory of project's contrib dir'''
            path = diff.b_path
            contrib_relpath = self.contrib_module_path
            try:
                return pathlib.Path(contrib_relpath) in \
                    pathlib.Path(path).parents
            except Exception:
                return False

        def is_appropriate_file_ext(diff):
            '''File change is a python file'''
            path = diff.b_path
            try:
                for ext in PullRequestFeatureValidator.APPROPRIATE_FILE_EXTS:
                    if path.endswith(ext):
                        return True
                return False
            except Exception:
                return False

        is_admissible = funcy.all_fn(
            is_appropriate_change_type,
            within_contrib_subdirectory,
            is_appropriate_file_ext,
        )

        for diff in self.file_diffs:
            if is_admissible(diff):
                self.file_diffs_admissible.append(diff)
                logger.debug(
                    'Categorized {file} as ADMISSIBLE'
                    .format(file=diff.b_path))
            else:
                self.file_diffs_inadmissible.append(diff)
                logger.debug(
                    'Categorized {file} as INADMISSIBLE'
                    .format(file=diff.b_path))

        logger.info('Admitted {} file(s) and rejected {} file(s)'.format(
            len(self.file_diffs_admissible),
            len(self.file_diffs_inadmissible)))

    def _validate_files(self):
        if self.file_diffs_inadmissible is None:
            raise UnexpectedValidationStateError(
                'File diffs have not been categorized.')

        result = len(self.file_diffs_inadmissible) == 0
        self.file_diffs_validation_result = result

    def _collect_features(self):
        if self.file_diffs_admissible is None:
            raise UnexpectedValidationStateError(
                'File diffs have not been collected.')

        logger.info('Collecting features...')

        self.features = []
        for diff in self.file_diffs_admissible:
            path = diff.b_path
            try:
                project_root = pathlib.Path(self.repo.working_tree_dir)
                modname = relpath_to_modname(path)
                modpath = str(project_root.joinpath(path))
                mod = import_module_at_path(modname, modpath)
            except ImportError:
                logger.info(
                    'Validation failure: failed to import module at {}'
                    .format(path))
                logger.exception('Exception details: ')
                self.features_validation_result = False
                continue

            features = get_contrib_features(mod)
            self.features.extend(features)

        logger.info('Collected {} feature(s)'.format(len(self.features)))

    def _validate_features(self):
        if self.features is None:
            raise UnexpectedValidationStateError(
                'Features have not been collected.')

        # if no features were added at all, reject
        if self.features is not None and len(self.features) == 0:
            logger.info('Failed to collect any new feature(s).')
            self.features_validation_result = False
            return

        # get small subset?
        X, y = subsample_data_for_validation(self.X, self.y)

        # validate
        feature_validator = FeatureValidator(X, y)
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
        if self.file_diffs_validation_result is None:
            raise UnexpectedValidationStateError(
                'File diffs have not been validated.')
        if self.features_validation_result is None:
            raise UnexpectedValidationStateError(
                'Feature changes have not been validated.')
        return (self.file_diffs_validation_result and
                self.features_validation_result)


def subsample_data_for_validation(X, y):
    # TODO
    return X, y
