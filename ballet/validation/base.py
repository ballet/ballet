from abc import ABCMeta, abstractmethod

from funcy import all_fn, ignore

from ballet.compat import pathlib
from ballet.contrib import get_contrib_features
from ballet.util import make_plural_suffix
from ballet.util.ci import TravisPullRequestBuildDiffer, can_use_travis_differ
from ballet.util.git import LocalPullRequestBuildDiffer
from ballet.util.log import logger
from ballet.util.mod import import_module_at_path, relpath_to_modname


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


class PostAcceptanceFeaturePerformanceEvaluator(FeaturePerformanceEvaluator):
    """Prune features after acceptance based on their performance"""

    @abstractmethod
    def prune(self):
        pass


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
        new_features, imported_okay = self._collect_features(
            file_diffs_admissible)

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
