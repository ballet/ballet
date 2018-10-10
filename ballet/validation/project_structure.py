import importlib.machinery

from funcy import all_fn, collecting, ignore, partial, post_processing

import ballet
from ballet.compat import pathlib
from ballet.contrib import get_contrib_features
from ballet.util import make_plural_suffix
from ballet.util.ci import TravisPullRequestBuildDiffer, can_use_travis_differ
from ballet.util.git import LocalPullRequestBuildDiffer
from ballet.util.log import logger, stacklog
from ballet.util.mod import import_module_at_path, relpath_to_modname
from ballet.validation.base import BaseValidator


def _log_collect_items(name, items):
    n = len(items)
    s = make_plural_suffix(items)
    logger.info('Collected {n} {name}{s}'.format(n=n, name=name, s=s))
    return items


class ChangeCollector:
    APPROPRIATE_CHANGE_TYPES = ['A']
    APPROPRIATE_FILE_EXTS = importlib.machinery.SOURCE_SUFFIXES

    def __init__(self, repo, pr_num, contrib_module_path):
        """Validate the features introduced in a proposed pull request.

        Args:
            repo (git.Repo): project repo
            pr_num (int, str): Pull request number
            contrib_module_path (str): Relative path to contrib module
        """
        self.repo = repo
        self.pr_num = str(pr_num)
        self.contrib_module_path = contrib_module_path

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
        new_feature_info = self._collect_feature_info(
            file_diffs_admissible)

        return (file_diffs, file_diffs_admissible, file_diffs_inadmissible,
                new_feature_info)

    @post_processing(partial(_log_collect_items, 'file'))
    @stacklog(logger.info, 'Collecting file changes')
    def _collect_file_diffs(self):
        file_diffs = self.differ.diff()

        # log results
        for i, file in enumerate(file_diffs):
            logger.debug('File {i}: {file}'.format(i=i, file=file))

        return file_diffs

    @stacklog(logger.info, 'Categorizing file changes')
    def _categorize_file_diffs(self, file_diffs):
        """Partition file changes into admissible and inadmissible changes"""
        # TODO move this into a new validator
        file_diffs_admissible = []
        file_diffs_inadmissible = []

        def is_appropriate_change_type(diff):
            """File change is an addition"""
            return diff.change_type in ChangeCollector.APPROPRIATE_CHANGE_TYPES

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
            for ext in ChangeCollector.APPROPRIATE_FILE_EXTS:
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

    @post_processing(partial(_log_collect_items, 'feature'))
    @collecting
    @stacklog(logger.info, 'Collecting info on newly-proposed features')
    def _collect_feature_info(self, file_diffs_admissible):
        project_root = pathlib.Path(self.repo.working_tree_dir)
        for diff in file_diffs_admissible:
            path = diff.b_path
            modname = relpath_to_modname(path)
            modpath = project_root.joinpath(path)
            importer = partial(import_module_at_path, modname, modpath)
            yield importer, modname, modpath


class FileChangeValidator(BaseValidator):

    def __init__(self, repo, pr_num, contrib_module_path):
        self.change_collector = ChangeCollector(
            repo, pr_num, contrib_module_path)

    def validate(self):
        _, _, inadmissible, _ = self.change_collector.collect_changes()
        return not inadmissible


def subsample_data_for_validation(X, y):
    return X, y


class FeatureApiValidator(BaseValidator):

    def __init__(self, repo, pr_num, contrib_module_path, X, y):
        self.change_collector = ChangeCollector(
            repo, pr_num, contrib_module_path)
        self.X, self.y = subsample_data_for_validation(X, y)

    def validate(self):
        """Collect and validate all new features"""

        _, _, _, new_feature_info = self.change_collector.collect_changes()

        for importer, modname, modpath in new_feature_info:
            features = []
            imported_okay = True
            try:
                mod = importer()
                features.extend(get_contrib_features(mod))
            except ImportError:
                logger.info(
                    'Validation failure: failed to import module at {}'
                    .format(modpath))
                logger.exception('Exception details: ')
                imported_okay = False

            if not imported_okay:
                return False

            # if no features were added at all, reject
            if not features:
                logger.info('Failed to collect any new features.')
                return False

            result = True
            for feature in features:
                success, failures = ballet.validation.feature_api.validate(
                    feature, self.X, self.y)
                if success:
                    logger.info(
                        'Feature is valid: {feature}'
                        .format(feature=feature))
                else:
                    logger.info(
                        'Feature is NOT valid: {feature}'
                        .format(feature=feature))
                    logger.debug(
                        'Failures in validation: {failures}'
                        .format(failures=failures))
                    result = False

            return result
