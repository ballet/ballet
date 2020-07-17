import pathlib
from types import ModuleType
from typing import (
    Callable, Collection, Iterator, List, NamedTuple, Optional, Tuple)

import git
from funcy import collecting, complement, lfilter, partial, post_processing
from stacklog import stacklog

from ballet.contrib import _collect_contrib_feature_from_module
from ballet.exc import (
    BalletError, FeatureCollectionError, NoFeaturesCollectedError)
from ballet.feature import Feature
from ballet.project import Project
from ballet.util import make_plural_suffix, whether_failures
from ballet.util.ci import TravisPullRequestBuildDiffer, can_use_travis_differ
from ballet.util.git import (
    Differ, LocalMergeBuildDiffer, LocalPullRequestBuildDiffer)
from ballet.util.log import logger
from ballet.util.mod import import_module_at_path, relpath_to_modname
from ballet.validation.base import FeaturePerformanceEvaluator
from ballet.validation.project_structure.checks import ProjectStructureCheck


def get_proposed_feature(project: Project):
    """Get the proposed feature

    The path of the proposed feature is determined by diffing the project
    against a comparison branch, such as master. The feature is then imported
    from that path and returned.

    Args:
        project: project info

    Raises:
        ballet.exc.BalletError: more than one feature collected
    """
    change_collector = ChangeCollector(project)
    collected_changes = change_collector.collect_changes()
    new_feature_info = collected_changes.new_feature_info

    n = len(new_feature_info)
    if n == 0:
        raise NoFeaturesCollectedError
    elif n == 1:
        importer, _, _ = new_feature_info[0]
        module = importer()
        feature = _collect_contrib_feature_from_module(module)
        return feature
    else:
        msg = 'Too many features collected (n={})'.format(n)
        raise FeatureCollectionError(msg)


def get_accepted_features(
    features: Collection[Feature],
    proposed_feature: Feature
) -> List[Feature]:
    """Deselect candidate features from list of all features

    Args:
        features: collection of all features in the ballet project: both
            accepted features and candidate ones that have not been accepted
        proposed_feature: candidate feature that has not been accepted

    Returns:
        list of features with the proposed feature not in it.

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


def _log_collect_items(name: str, items: Collection):
    n = len(items)
    s = make_plural_suffix(items)
    logger.info('Collected {n} {name}{s}'.format(n=n, name=name, s=s))
    return items


NewFeatureInfo = Tuple[Callable[..., ModuleType], str, str]


class CollectedChanges(NamedTuple):
    file_diffs: git.DiffIndex
    candidate_feature_diffs: List[git.Diff]
    valid_init_diffs: List[git.Diff]
    inadmissible_diffs: List[git.Diff]
    new_feature_info: List[NewFeatureInfo]


class ChangeCollector:
    """Validate the features introduced in a proposed pull request.

    Args:
        project: project info
        differ: specific differ to use; if not provided, will be determined
            automatically from the environment
    """

    def __init__(self, project: Project, differ: Optional[Differ] = None):
        self.project = project

        self.differ: Differ
        if differ is not None:
            self.differ = differ
        else:
            pr_num = self.project.pr_num
            repo = self.project.repo
            if pr_num is None:
                self.differ = LocalMergeBuildDiffer(repo)
            elif can_use_travis_differ():
                self.differ = TravisPullRequestBuildDiffer(pr_num)
            else:
                self.differ = LocalPullRequestBuildDiffer(pr_num, repo)

    def collect_changes(self) -> CollectedChanges:
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
    def _collect_file_diffs(self) -> git.DiffIndex:
        file_diffs = self.differ.diff()

        # log results
        for i, file in enumerate(file_diffs):
            logger.debug('File {i}: {file}'.format(i=i, file=file))

        return file_diffs

    @stacklog(logger.info, 'Categorizing file changes')
    def _categorize_file_diffs(
        self, file_diffs: git.DiffIndex
    ) -> Tuple[List[git.Diff], List[git.Diff], List[git.Diff]]:
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
    def _collect_feature_info(
        self, candidate_feature_diffs: List[git.Diff]
    ) -> Iterator[NewFeatureInfo]:
        """Collect feature info

        Args:
            candidate_feature_diffs: list of Diffs corresponding to admissible
                file changes compared to comparison ref

        Returns:
            list of tuple of importer, module name, and module path. The
                "importer" is a callable that returns a module
        """

        # the directory containing ballet.yml
        project_root = self.project.path

        # the directory containing the package
        try:
            package_path = self.project.package.__path__[0]  # type: ignore  # mypy issue #1422  # noqa E501
            package_root = pathlib.Path(package_path).parent
        except (AttributeError, IndexError):
            logger.debug("Couldn't get package root, will try to recover",
                         exc_info=True)
            package_root = project_root

        for diff in candidate_feature_diffs:
            path = diff.b_path
            relpath = project_root.joinpath(path).relative_to(package_root)
            modname = relpath_to_modname(relpath)
            modpath = project_root.joinpath(path)
            importer = partial(import_module_at_path, modname, modpath)
            yield importer, modname, modpath


def subsample_data_for_validation(X, y):
    return X, y


@whether_failures
def check_from_class(check_class: type, obj, *checker_args, **checker_kwargs):
    for Checker in check_class.__subclasses__():
        check = Checker(*checker_args, **checker_kwargs).do_check
        success = check(obj)
        if not success:
            name = Checker.__name__
            yield name


class RandomFeaturePerformanceEvaluator(FeaturePerformanceEvaluator):

    def __init__(self, *args, p=0.3, seed=None):
        super().__init__(*args)
        self.p = p
        self.seed = seed

    def __str__(self):
        return '{str}: p={p}, seed={seed}'.format(
            str=super().__str__(), p=self.p, seed=self.seed)
