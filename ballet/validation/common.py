import pathlib
from types import ModuleType
from typing import (
    Callable, Collection, Iterator, List, NamedTuple, Optional, Tuple,)

import git
from funcy import (
    collecting, complement, lfilter, partial, post_processing, silent,)
from stacklog import stacklog

from ballet.contrib import _collect_contrib_feature_from_module
from ballet.exc import (
    BalletError, FeatureCollectionError, NoFeaturesCollectedError,)
from ballet.feature import Feature
from ballet.project import Project
from ballet.util import make_plural_suffix
from ballet.util.ci import TravisPullRequestBuildDiffer, can_use_travis_differ
from ballet.util.git import (
    Differ, LocalMergeBuildDiffer, LocalPullRequestBuildDiffer, NoOpDiffer,
    can_use_local_differ, can_use_local_merge_differ,)
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
        raise FeatureCollectionError(f'Too many features collected (n={n})')


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
            f'Unexpected condition (n_features={len(features)}, '
            f'n_result={len(result)})')


def _log_collect_items(name: str, items: Collection):
    n = len(items)
    s = make_plural_suffix(items)
    logger.info(f'Collected {n} {name}{s}')
    return items


class NewFeatureInfo(NamedTuple):
    importer: Callable[[], ModuleType]
    modname: str
    modpath: str


class CollectedChanges(NamedTuple):
    file_diffs: git.DiffIndex
    candidate_feature_diffs: List[git.Diff]
    valid_init_diffs: List[git.Diff]
    inadmissible_diffs: List[git.Diff]
    new_feature_info: List[NewFeatureInfo]


def detect_differ(repo):
    if can_use_travis_differ(repo):
        return TravisPullRequestBuildDiffer(repo)
    elif can_use_local_merge_differ(repo):
        return LocalMergeBuildDiffer(repo)
    elif can_use_local_differ(repo):
        return LocalPullRequestBuildDiffer(repo)
    else:
        return NoOpDiffer(repo)


class ChangeCollector:
    """Validate the features introduced in a proposed change set.

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
            self.differ = detect_differ(self.project.repo)

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
            logger.debug(f'File {i}: {file}')

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
            valid, failures, _ = check_from_class(
                ProjectStructureCheck, diff, self.project)
            if valid:
                if pathlib.Path(diff.b_path).parts[-1] != '__init__.py':
                    candidate_feature_diffs.append(diff)
                    logger.debug(
                        f'Categorized {diff.b_path} as '
                        'CANDIDATE FEATURE MODULE')
                else:
                    valid_init_diffs.append(diff)
                    logger.debug(
                        f'Categorized {diff.b_path} as VALID INIT MODULE')
            else:
                inadmissible_files.append(diff)
                logger.debug(
                    f'Categorized {diff.b_path} as INADMISSIBLE; '
                    f'failures were {failures}')

        logger.info(
            'Admitted {n1} candidate feature{s1} '
            'and {n2} __init__ module{s2} '
            'and rejected {n3} file{s3}'
            .format(n1=len(candidate_feature_diffs),
                    s1=make_plural_suffix(candidate_feature_diffs),
                    n2=len(valid_init_diffs),
                    s2=make_plural_suffix(valid_init_diffs),
                    n3=len(inadmissible_files),
                    s3=make_plural_suffix(inadmissible_files)))

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
            yield NewFeatureInfo(importer, modname, modpath)


def subsample_data_for_validation(*args):
    return args


def get_subclasses(cls):
    return cls.__subclasses__()


def check_from_class(check_class: type, obj, *checker_args, **checker_kwargs):
    failures = []
    advice = []
    for Checker in get_subclasses(check_class):
        checker = Checker(*checker_args, **checker_kwargs)
        check = checker.do_check
        success = check(obj)
        if not success:
            failures.append(Checker.__name__)
            advice_item = silent(checker.give_advice)(obj)
            advice.append(advice_item)

    valid = not failures
    return valid, failures, advice


class RandomFeaturePerformanceEvaluator(FeaturePerformanceEvaluator):

    def __init__(self, *args, p=0.3, seed=None):
        super().__init__(*args)
        self.p = p
        self.seed = seed

    def __str__(self):
        return f'{super().__str__()}: p={self.p}, seed={self.seed}'
