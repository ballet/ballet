import copy
from abc import ABCMeta, abstractmethod

import git
from funcy import all_fn, ignore, isa, iterable

from ballet.compat import pathlib
from ballet.contrib import get_contrib_features
from ballet.exc import (
    FeatureRejected, InvalidFeatureApi, InvalidProjectStructure)
from ballet.feature import Feature
from ballet.util import make_plural_suffix, validation_check
from ballet.util.ci import (
    TravisPullRequestBuildDiffer, can_use_travis_differ, detect_target_type,
    get_travis_pr_num)
from ballet.util.git import LocalPullRequestBuildDiffer
from ballet.util.log import logger
from ballet.util.mod import import_module_at_path, relpath_to_modname


def get_proposed_features(project):
    repo = git.Repo(project['here'](), search_parent_directories=True)
    pr_num = get_travis_pr_num()
    contrib_module_path = project['get']('contrib', 'module_path')
    X_df, y_df = project['load_data']()
    validator = ProjectStructureValidator(
        repo, pr_num, contrib_module_path, X_df, y_df)
    _, _, _, new_features, _ = validator.collect_changes()
    return new_features


def check_project_structure(project):
    repo = git.Repo(project['here'](), search_parent_directories=True)
    pr_num = get_travis_pr_num()
    contrib_module_path = project['get']('contrib', 'module_path')
    X_df, y_df = project['load_data']()
    validator = ProjectStructureValidator(
        repo, pr_num, contrib_module_path, X_df, y_df)
    result = validator.validate()
    if not result:
        raise InvalidProjectStructure


def validate_feature_api(project):
    repo = git.Repo(project['here'](), search_parent_directories=True)
    pr_num = get_travis_pr_num()
    contrib_module_path = project['get']('contrib', 'module_path')
    X_df, y_df = project['load_data']()
    validator = FeatureApiValidator(
        repo, pr_num, contrib_module_path, X_df, y_df)
    result = validator.validate()
    if not result:
        raise InvalidFeatureApi


def evaluate_feature_performance(project):
    X_df, y_df = project['load_data']()
    features = project['get_contrib_features']()
    evaluator = FeatureRelevanceEvaluator(X_df, y_df, features)
    proposed_features = get_proposed_features(project)
    accepted = evaluator.judge(proposed_features)
    if not accepted:
        raise FeatureRejected


def prune_existing_features(project):
    X_df, y_df = project['load_data']()
    features = project['get_contrib_features']()
    evaluator = FeatureRedundancyEvaluator(X_df, y_df, features)
    redundant_features = evaluator.prune()

    # propose removal
    for feature in redundant_features:
        pass


class FeaturePerformanceEvaluator(metaclass=ABCMeta):
    """Evaluate the performance of features from an ML point-of-view"""

    def __init__(self, X_df, y_df, features):
        self.X_df = X_df
        self.y_df = y_df
        self.features = features


class PreAcceptanceFeaturePerformanceEvaluator(FeaturePerformanceEvaluator):
    """Accept/reject a feature to the project based on its performance"""

    @abstractmethod
    def judge(self, feature):
        pass


class FeatureRelevanceEvaluator(PreAcceptanceFeaturePerformanceEvaluator):
    """Accept a feature if it is correlated to the target"""

    def judge(self, feature):
        return False


class PostAcceptanceFeaturePerformanceEvaluator(FeaturePerformanceEvaluator):
    """Prune features after acceptance based on their performance"""

    @abstractmethod
    def prune(self):
        pass


class FeatureRedundancyEvaluator(PreAcceptanceFeaturePerformanceEvaluator):
    """Remove a feature if it is conditionally independent of the target

    Let Sk be the set of subsets of features of size less than or equal to k.
    A feature Xi is redundant if it is independent of the target, conditional
    on some S in Sk. If a feature is redundant, it is removed from the feature
    matrix.
    """

    def prune(self, k=4):
        return []


class SingleFeatureApiValidator:
    """Validate that a feature confirms to the feature API"""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def validate(self, feature):
        """Validate the feature"""
        failures = []
        for check, name in self._get_all_checks():
            success = check(feature)
            if not success:
                failures.append(name)

        result = not failures

        return result, failures

    @validation_check
    def _is_feature(self, feature):
        assert isinstance(feature, Feature)

    @validation_check
    def _has_correct_input_type(self, feature):
        """Check that `input` is a string or iterable of string"""
        input = feature.input
        is_str = isa(str)
        is_nested_str = all_fn(
            iterable, lambda x: all(map(is_str, x)))
        assert is_str(input) or is_nested_str(input)

    @validation_check
    def _has_transformer_interface(self, feature):
        assert hasattr(feature.transformer, 'fit')
        assert hasattr(feature.transformer, 'transform')
        assert hasattr(feature.transformer, 'fit_transform')

    @validation_check
    def _can_make_mapper(self, feature):
        feature.as_dataframe_mapper()

    @validation_check
    def _can_fit(self, feature):
        mapper = feature.as_dataframe_mapper()
        mapper.fit(self.X, self.y)

    @validation_check
    def _can_transform(self, feature):
        mapper = feature.as_dataframe_mapper()
        mapper.fit(self.X, self.y)
        mapper.transform(self.X)

    @validation_check
    def _can_fit_transform(self, feature):
        mapper = feature.as_dataframe_mapper()
        mapper.fit_transform(self.X, self.y)

    @validation_check
    def _has_correct_output_dimensions(self, feature):
        mapper = feature.as_dataframe_mapper()
        X = mapper.fit_transform(self.X, self.y)

        assert self.X.shape[0] == X.shape[0]

    @validation_check
    def _can_deepcopy(self, feature):
        copy.deepcopy(feature)

    def _get_all_checks(self):
        for method_name in self.__dir__():
            method = getattr(self, method_name)
            if hasattr(method, 'is_check') and method.is_check:
                name = method.__name__
                if name.startswith('_'):
                    name = name[1:]
                yield (method, name)


class ProjectStructureValidator:
    APPROPRIATE_CHANGE_TYPES = ['A']
    APPROPRIATE_FILE_EXTS = ['.py']

    def __init__(self, repo, pr_num, contrib_module_path, X, y):
        """Validate the features introduced in a proposed pull request.

        Args:
            repo (git.Repo): project repo
            pr_num (int, str): Pull request number
            contrib_module_path (str): Relative path to contrib module
            X (array-like): Example X array-like
            y (array-like): Example y array-like
        """
        self.repo = repo
        self.pr_num = str(pr_num)
        self.contrib_module_path = contrib_module_path
        self.X = X
        self.y = y

        if can_use_travis_differ():
            self.differ = TravisPullRequestBuildDiffer(self.pr_num)
        else:
            self.differ = LocalPullRequestBuildDiffer()

    def collect_changes(self):
        """Collect file and feature changes

        Steps
        1. Collects the files that have changed in this pull request as
           compared to a comparison branch.
        2. Categorize these file changes into admissible or inadmissible file
           changes. Admissible file changes solely contribute python files to
           the contrib subdirectory.
        3. Collect features from admissible new files.
        """

        file_diffs = self._collect_file_diffs()
        file_diffs_admissible, file_diffs_inadmissible = \
            self._categorize_file_diffs(file_diffs)
        new_features, imported_okay = self._collect_features(file_diffs_admissible)

        return (file_diffs, file_diffs_admissible, file_diffs_inadmissible,
                new_features, imported_okay)

    def _collect_file_diffs(self):
        logger.info('Collecting file changes...')

        file_diffs = self.differ.diff()

        # log results
        for i, file in enumerate(file_diffs):
            logger.debug('File {i}: {file}'.format(i=i, file=file))

        n = len(file_diffs)
        s = make_plural_suffix(file_diffs)
        logger.info('Collected {n} file{s}'.format(n=n, s=s))

        return file_diffs

    def _categorize_file_diffs(self, file_diffs):
        """Partition file changes into admissible and inadmissible changes"""
        logger.info('Categorizing file changes...')

        file_diffs_admissible = []
        file_diffs_inadmissible = []

        def is_appropriate_change_type(diff):
            """File change is an addition"""
            return diff.change_type in \
                ProjectStructureValidator.APPROPRIATE_CHANGE_TYPES

        @ignore(Exception, default=False)
        def within_contrib_subdirectory(diff):
            """File addition is a subdirectory of project's contrib dir"""
            path = diff.b_path
            contrib_relpath = self.contrib_module_path
            return pathlib.Path(contrib_relpath) in pathlib.Path(path).parents

        @ignore(Exception, default=False)
        def is_appropriate_file_ext(diff):
            """File change is a python file"""
            path = diff.b_path
            for ext in ProjectStructureValidator.APPROPRIATE_FILE_EXTS:
                if path.endswith(ext):
                    return True

            return False

        is_admissible = all_fn(
            is_appropriate_change_type,
            within_contrib_subdirectory,
            is_appropriate_file_ext,
        )

        for diff in file_diffs:
            if is_admissible(diff):
                file_diffs_admissible.append(diff)
                logger.debug(
                    'Categorized {file} as ADMISSIBLE'
                    .format(file=diff.b_path))
            else:
                file_diffs_inadmissible.append(diff)
                logger.debug(
                    'Categorized {file} as INADMISSIBLE'
                    .format(file=diff.b_path))

        logger.info('Admitted {} file{} and rejected {} file{}'.format(
            len(file_diffs_admissible),
            make_plural_suffix(file_diffs_admissible),
            len(file_diffs_inadmissible),
            make_plural_suffix(file_diffs_inadmissible)))

        return file_diffs_admissible, file_diffs_inadmissible

    def _collect_features(self, file_diffs_admissible):

        logger.info('Collecting newly-proposed features...')

        new_features = []
        imported_okay = True
        for diff in file_diffs_admissible:
            path = diff.b_path
            project_root = pathlib.Path(self.repo.working_tree_dir)
            modname = relpath_to_modname(path)
            modpath = project_root.joinpath(path)
            try:
                mod = import_module_at_path(modname, modpath)
            except ImportError:
                logger.info(
                    'Validation failure: failed to import module at {}'
                    .format(path))
                logger.exception('Exception details: ')
                imported_okay = False
            else:
                new_features.extend(get_contrib_features(mod))

        n = len(new_features)
        s = make_plural_suffix(new_features)
        logger.info('Collected {n} feature{s}'.format(n=n, s=s))

        return new_features, imported_okay

    def validate(self):
        raise NotImplementedError


class FileChangeValidator(ProjectStructureValidator):

    def validate(self):
        _, _, inadmissible, _, imported_okay = self.collect_changes()
        return not inadmissible and imported_okay


class FeatureApiValidator(ProjectStructureValidator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.X, self.y = subsample_data_for_validation(self.X, self.y)

    def validate(self):
        """Collect and validate all new features"""

        diffs, admissible, inadmissible, new_features, imported_okay = self.collect_changes()

        # if no features were added at all, reject
        if not new_features:
            logger.info('Failed to collect any new features.')
            return False

        feature_validator = SingleFeatureApiValidator(self.X, self.y)

        # validate
        okay = True
        for feature in new_features:
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
                okay = False

        return okay


def subsample_data_for_validation(X, y):
    # TODO
    return X, y


def main(project, target_type=None):
    if target_type is None:
        target_type = detect_target_type()

    if target_type == 'project_structure_validation':
        check_project_structure(project)
    elif target_type == 'feature_api_validation':
        validate_feature_api(project)
    elif target_type == 'pre_acceptance_feature_evaluation':
        evaluate_feature_performance(project)
    elif target_type == 'post_acceptance_feature_evaluation':
        prune_existing_features(project)
    else:
        raise NotImplementedError
