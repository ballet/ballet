import importlib.machinery
from abc import ABCMeta, abstractmethod
from collections import namedtuple

from funcy import collecting, ignore, partial, post_processing, re_test

import ballet
import ballet.validation.feature_api
from ballet.compat import pathlib
from ballet.contrib import _get_contrib_features
from ballet.util import make_plural_suffix, whether_failures
from ballet.util.ci import TravisPullRequestBuildDiffer, can_use_travis_differ
from ballet.util.fs import isemptyfile
from ballet.util.git import LocalPullRequestBuildDiffer
from ballet.util.log import logger, stacklog
from ballet.util.mod import import_module_at_path, relpath_to_modname
from ballet.validation.base import BaseValidator

FEATURE_MODULE_NAME_REGEX = r'feature_[a-zA-Z0-9_]+\.\w+'
SUBPACKAGE_NAME_REGEX = r'user_[a-zA-Z0-9_]+'


def _log_collect_items(name, items):
    n = len(items)
    s = make_plural_suffix(items)
    logger.info('Collected {n} {name}{s}'.format(n=n, name=name, s=s))
    return items


@whether_failures
def is_admissible(diff, project):
    for Checker in DiffCheck.__subclasses__():
        check = Checker(project).do_check
        name = Checker.__name__
        success = check(diff)
        if not success:
            yield name


class DiffCheck(metaclass=ABCMeta):

    def __init__(self, project):
        self.project = project

    @ignore(Exception, default=False)
    def do_check(self, diff):
        return self.check(diff)

    @abstractmethod
    def check(self, diff):
        pass


class IsAdditionCheck(DiffCheck):

    def check(self, diff):
        return diff.change_type == 'A'


class IsPythonSourceCheck(DiffCheck):

    def check(self, diff):
        path = diff.b_path
        return any(
            path.endswith(ext)
            for ext in importlib.machinery.SOURCE_SUFFIXES
        )


class WithinContribCheck(DiffCheck):

    def check(self, diff):
        path = diff.b_path
        contrib_path = self.project.contrib_module_path
        return pathlib.Path(contrib_path) in pathlib.Path(path).parents


def relative_to_contrib(diff, project):
    """Compute relative path of changed file to contrib dir

    Args:
        diff (git.diff.Diff): file diff
        project (Project): project

    Returns:
        Path
    """
    path = pathlib.Path(diff.b_path)
    contrib_path = project.contrib_module_path
    return path.relative_to(contrib_path)


class SubpackageNameCheck(DiffCheck):

    def check(self, diff):
        relative_path = relative_to_contrib(diff, self.project)
        subpackage_name = relative_path.parts[0]
        return re_test(SUBPACKAGE_NAME_REGEX, subpackage_name)


class RelativeNameDepthCheck(DiffCheck):

    def check(self, diff):
        relative_path = relative_to_contrib(diff, self.project)
        return len(relative_path.parts) == 2


class ModuleNameCheck(DiffCheck):

    def check(self, diff):
        filename = pathlib.Path(diff.b_path).parts[-1]
        is_valid_feature_module_name = re_test(
            FEATURE_MODULE_NAME_REGEX, filename)
        is_valid_init_module_name = filename == '__init__.py'
        return is_valid_feature_module_name or is_valid_init_module_name


class IfInitModuleThenIsEmptyCheck(DiffCheck):

    def check(self, diff):
        path = pathlib.Path(diff.b_path)
        filename = path.parts[-1]
        if filename == '__init__.py':
            abspath = self.project.path.joinpath(path)
            return isemptyfile(abspath)
        else:
            return True


CollectedChanges = namedtuple(
    'CollectedChanges',
    'file_diffs candidate_feature_diffs valid_init_diffs inadmissible_diffs '
    'new_feature_info')


class ChangeCollector:

    def __init__(self, project):
        """Validate the features introduced in a proposed pull request.

        Args:
            repo (git.Repo): project repo
            pr_num (int, str): Pull request number
            contrib_module_path (str): Relative path to contrib module
        """
        self.project = project
        self.repo = project.repo
        self.pr_num = str(project.pr_num)
        self.contrib_module_path = project.contrib_module_path

        if can_use_travis_differ():
            self.differ = TravisPullRequestBuildDiffer(self.pr_num)
        else:
            self.differ = LocalPullRequestBuildDiffer(self.pr_num, self.repo)

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
        candidate_feature_diffs, valid_init_diffs, inadmissible_diffs = \
            self._categorize_file_diffs(file_diffs)
        new_feature_info = self._collect_feature_info(candidate_feature_diffs)

        return CollectedChanges(
            file_diffs, candidate_feature_diffs, valid_init_diffs,
            inadmissible_diffs, new_feature_info)

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
        candidate_feature_diffs = []
        valid_init_diffs = []
        inadmissible_files = []

        for diff in file_diffs:
            admissible, failures = is_admissible(diff, self.project)
            if admissible:
                if pathlib.Path(diff.b_path).parts[-1] != '__init__.py':
                    candidate_feature_diffs.append(diff)
                    logger.debug(
                        'Categorized {file} as CANDIDATE FEATURE MODULE'
                        .format(file=diff.b_path))
                else:
                    valid_init_diffs.append(diff)
                    logger.debug(
                        'Categorized {file} as VALID INIT MODULE'
                        .format(file=diff.b_path))
            else:
                inadmissible_files.append(diff)
                logger.debug(
                    'Categorized {file} as INADMISSIBLE; '
                    'failures were {failures}'
                    .format(file=diff.b_path, failures=failures))

        logger.info(
            'Admitted {} candidate feature{} '
            'and {} __init__ module{} '
            'and rejected {} file{}'
            .format(len(candidate_feature_diffs),
                    make_plural_suffix(candidate_feature_diffs),
                    len(valid_init_diffs),
                    make_plural_suffix(valid_init_diffs),
                    len(inadmissible_files),
                    make_plural_suffix(inadmissible_files)))

        return candidate_feature_diffs, valid_init_diffs, inadmissible_files

    @post_processing(partial(_log_collect_items, 'feature'))
    @collecting
    @stacklog(logger.info, 'Collecting info on newly-proposed features')
    def _collect_feature_info(self, candidate_feature_diffs):
        """Collect feature info

        Args:
            candidate_feature_diffs (List[git.diff.Diff]): list of Diffs
                corresponding to admissible file changes compared to comparison
                ref

        Returns:
            List[Tuple]: list of tuple of importer, module name, and module
                path. The "importer" is a callable that returns a module
        """
        project_root = pathlib.Path(self.repo.working_tree_dir)
        for diff in candidate_feature_diffs:
            path = diff.b_path
            modname = relpath_to_modname(path)
            modpath = project_root.joinpath(path)
            importer = partial(import_module_at_path, modname, modpath)
            yield importer, modname, modpath


class FileChangeValidator(BaseValidator):

    def __init__(self, project):
        self.change_collector = ChangeCollector(project)

    def validate(self):
        collected_changes = self.change_collector.collect_changes()
        return not collected_changes.inadmissible_diffs


def subsample_data_for_validation(X, y):
    return X, y


class FeatureApiValidator(BaseValidator):

    def __init__(self, project):
        self.change_collector = ChangeCollector(project)

        X, y = project.load_data()
        self.X, self.y = subsample_data_for_validation(X, y)

    def validate(self):
        """Collect and validate all new features"""

        collected_changes = self.change_collector.collect_changes()

        for importer, modname, modpath in collected_changes.new_feature_info:
            features = []
            imported_okay = True
            try:
                mod = importer()
                features.extend(_get_contrib_features(mod))
            except (ImportError, SyntaxError):
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
                        'Feature {feature!r} is valid'
                        .format(feature=feature))
                else:
                    logger.info(
                        'Feature {feature!r} is NOT valid; '
                        'failures were {failures}'
                        .format(feature=feature, failures=failures))
                    result = False

            return result
