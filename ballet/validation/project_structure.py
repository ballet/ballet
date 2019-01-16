from collections import namedtuple

from funcy import collecting, partial, post_processing

from ballet.compat import pathlib
from ballet.contrib import _get_contrib_features
from ballet.util import make_plural_suffix
from ballet.util.ci import TravisPullRequestBuildDiffer, can_use_travis_differ
from ballet.util.git import LocalPullRequestBuildDiffer
from ballet.util.log import logger, stacklog
from ballet.util.mod import import_module_at_path, relpath_to_modname
from ballet.validation.base import BaseValidator, check_from_class
from ballet.validation.diff_checks import DiffCheck
from ballet.validation.feature_api_checks import FeatureApiCheck


def _log_collect_items(name, items):
    n = len(items)
    s = make_plural_suffix(items)
    logger.info('Collected {n} {name}{s}'.format(n=n, name=name, s=s))
    return items


CollectedChanges = namedtuple(
    'CollectedChanges',
    'file_diffs candidate_feature_diffs valid_init_diffs inadmissible_diffs '
    'new_feature_info')


class ChangeCollector:
    """Validate the features introduced in a proposed pull request.

    Args:
        repo (git.Repo): project repo
        pr_num (int, str): Pull request number
        contrib_module_path (str): Relative path to contrib module
    """

    def __init__(self, project):
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
            valid, failures = check_from_class(
                DiffCheck, diff, self.project)
            if valid:
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
                valid, failures = check_from_class(
                    FeatureApiCheck, feature, self.X, self.y)
                if valid:
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
