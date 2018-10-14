import importlib.machinery
from abc import ABCMeta, abstractmethod

from funcy import collecting, ignore, partial, post_processing, re_test

import ballet
import ballet.validation.feature_api
from ballet.compat import pathlib
from ballet.contrib import _get_contrib_features
from ballet.util import make_plural_suffix, whether_failures
from ballet.util.ci import TravisPullRequestBuildDiffer, can_use_travis_differ
from ballet.util.git import LocalPullRequestBuildDiffer
from ballet.util.log import logger, stacklog
from ballet.util.mod import import_module_at_path, relpath_to_modname
from ballet.validation.base import BaseValidator

FEATURE_MODULE_NAME_REGEX = r'feature_[a-zA-Z0-9]'
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
        path = diff.path
        contrib_path = self.project.contrib_module_path
        return pathlib.Path(contrib_path) in pathlib.Path(path).parents


def relative_to_contrib(diff, project):
    path = pathlib.Path(diff.path)
    contrib_path = project.contrib_module_path
    return path.relative_to(contrib_path)


class SubpackageNameTest(DiffCheck):

    def check(self, diff):
        relative_path = relative_to_contrib(diff, self.project)
        subpackage_name = relative_path.parts[0]
        return re_test(SUBPACKAGE_NAME_REGEX, subpackage_name)


class FeatureModuleNameTest(DiffCheck):

    def check(self, diff):
        relative_path = relative_to_contrib(diff, self.project)
        feature_module_name = relative_path.parts[-1]
        return re_test(FEATURE_MODULE_NAME_REGEX, feature_module_name)


class RelativeNameDepthCheck(DiffCheck):

    def check(self, diff):
        relative_path = relative_to_contrib(diff, self.project)
        return len(relative_path.parts) == 2


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
        (file_diffs_admissible,
         file_diffs_inadmissible) = self._categorize_file_diffs(file_diffs)
        new_feature_info = self._collect_feature_info(file_diffs_admissible)

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

        for diff in file_diffs:
            admissible, failures = is_admissible(diff, self.project)
            if admissible:
                file_diffs_admissible.append(diff)
                logger.debug(
                    'Categorized {file} as ADMISSIBLE'
                    .format(file=diff.b_path))
            else:
                file_diffs_inadmissible.append(diff)
                logger.debug(
                    'Categorized {file} as INADMISSIBLE; '
                    'failures were {failures}'
                    .format(file=diff.b_path, failures=failures))

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
        """Collect feature info

        Args:
            file_diffs_admissible (List[git.diff.Diff]): list of Diffs
                corresponding to admissible file changes compared to comparison
                ref

        Returns:
            List[Tuple]: list of tuple of importer, module name, and module
                path. The "importer" is a callable that returns a module
        """
        project_root = pathlib.Path(self.repo.working_tree_dir)
        for diff in file_diffs_admissible:
            path = diff.b_path
            modname = relpath_to_modname(path)
            modpath = project_root.joinpath(path)
            importer = partial(import_module_at_path, modname, modpath)
            yield importer, modname, modpath


class FileChangeValidator(BaseValidator):

    def __init__(self, project):
        self.change_collector = ChangeCollector(project)

    def validate(self):
        _, _, inadmissible, new_feature_info = \
            self.change_collector.collect_changes()
        return not inadmissible


def subsample_data_for_validation(X, y):
    return X, y


class FeatureApiValidator(BaseValidator):

    def __init__(self, project):
        self.change_collector = ChangeCollector(project)

        X, y = project.load_data()
        self.X, self.y = subsample_data_for_validation(X, y)

    def validate(self):
        """Collect and validate all new features"""

        _, _, _, new_feature_info = self.change_collector.collect_changes()

        for importer, modname, modpath in new_feature_info:
            features = []
            imported_okay = True
            try:
                mod = importer()
                features.extend(_get_contrib_features(mod))
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
