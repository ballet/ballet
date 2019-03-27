from collections import namedtuple

from funcy import collecting, complement, lfilter, partial, post_processing

from ballet.compat import pathlib
from ballet.contrib import _get_contrib_feature_from_module
from ballet.util import make_plural_suffix, one_or_raise
from ballet.util.ci import TravisPullRequestBuildDiffer, can_use_travis_differ
from ballet.exc import BalletError
from ballet.util.git import LocalPullRequestBuildDiffer
from ballet.util.log import logger, stacklog
from ballet.util.mod import import_module_at_path, relpath_to_modname
from ballet.validation.base import check_from_class
from ballet.validation.project_structure.checks import ProjectStructureCheck


def get_proposed_feature(project):
    """Get the proposed feature

    The path of the proposed feature is determined by diffing the project
    against a comparison branch, such as master. The feature is then imported
    from that path and returned.

    Args:
        project (ballet.project.Project): project info

    Raises:
        ballet.exc.BalletError: more than one feature collected
    """
    change_collector = ChangeCollector(project)
    collected_changes = change_collector.collect_changes()
    try:
        new_feature_info = one_or_raise(collected_changes.new_feature_info)
        importer, _, _ = new_feature_info
    except ValueError:
        raise BalletError('Too many features collected')
    module = importer()
    feature = _get_contrib_feature_from_module(module)
    return feature


def get_accepted_features(features, proposed_feature):
    """Deselect candidate features from list of all features

    Args:
        features (List[Feature]): collection of all features in the ballet
            project: both accepted features and candidate ones that have not
            been accepted
        proposed_feature (Feature): candidate feature that has not been
            accepted

    Returns:
        List[Feature]: list of features with the proposed feature not in it.

    Raises:
        ballet.exc.BalletError: Could not deselect exactly the proposed
            feature.
    """
    def eq(feature):
        """Features are equal if they have the same source

        At least in this implementation...
        """
        return feature.source == proposed_feature.source

    # deselect features that match the proposed feature
    result = lfilter(complement(eq), features)

    if len(features) - len(result) == 1:
        return result
    elif len(result) == len(features):
        raise BalletError(
            'Did not find match for proposed feature within \'contrib\'')
    else:
        raise BalletError(
            'Unexpected condition (n_features={}, n_result={})'
                .format(len(features), len(result)))




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
                ProjectStructureCheck, diff, self.project)
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


def subsample_data_for_validation(X, y):
    return X, y
