import os
from typing import Dict, Iterable, Optional

import git

from ballet.exc import UnexpectedTravisEnvironmentError
from ballet.util import falsy, truthy
from ballet.util.git import (
    PullRequestBuildDiffer, get_diff_endpoints_from_commit_range,)
from ballet.util.log import logger


def in_travis() -> bool:
    """Check if we are in Travis right now"""
    return len(get_travis_env_vars()) > 0


def get_travis_env_or_fail(name: str) -> str:
    if name in os.environ:
        return os.environ[name]
    else:
        # dump_travis_env_vars()
        raise UnexpectedTravisEnvironmentError(
            f'Missing TRAVIS environment variable: {name}')


def ensure_expected_travis_env_vars(names: Iterable[str]):
    if not set(names).issubset(set(os.environ.keys())):
        raise UnexpectedTravisEnvironmentError


def get_travis_env_vars() -> Dict[str, str]:
    return {
        k: v for k, v in os.environ.items()
        if k.startswith('TRAVIS')
    }


def dump_travis_env_vars():
    logger.info(repr(get_travis_env_vars()))


# TODO delete
def get_travis_pr_num() -> Optional[int]:
    """Return the PR number if the job is a pull request, None otherwise

    Returns:
        int

    See also:
        - <https://docs.travis-ci.com/user/environment-variables/#default-environment-variables>
    """  # noqa E501
    try:
        travis_pull_request = get_travis_env_or_fail('TRAVIS_PULL_REQUEST')
        if falsy(travis_pull_request):
            return None
        else:
            try:
                return int(travis_pull_request)
            except ValueError:
                return None
    except UnexpectedTravisEnvironmentError:
        return None


def is_travis_pr() -> bool:
    """Check if the current job is a pull request build"""
    return get_travis_pr_num() is not None


# TODO delete
def get_travis_branch() -> Optional[str]:
    """Get current branch per Travis environment variables

    If travis is building a PR, then TRAVIS_PULL_REQUEST is truthy and the
    name of the branch corresponding to the PR is stored in the
    TRAVIS_PULL_REQUEST_BRANCH environment variable. Else, the name of the
    branch is stored in the TRAVIS_BRANCH environment variable.

    See also:
        - <https://docs.travis-ci.com/user/environment-variables/#default-environment-variables>
    """  # noqa E501
    try:
        travis_pull_request = get_travis_env_or_fail('TRAVIS_PULL_REQUEST')
        if truthy(travis_pull_request):
            travis_pull_request_branch = get_travis_env_or_fail(
                'TRAVIS_PULL_REQUEST_BRANCH')
            return travis_pull_request_branch
        else:
            travis_branch = get_travis_env_or_fail('TRAVIS_BRANCH')
            return travis_branch
    except UnexpectedTravisEnvironmentError:
        return None


def can_use_travis_differ(repo: Optional[git.Repo]) -> bool:
    """Check if the required travis env vars are set for the travis differ"""
    try:
        ensure_expected_travis_env_vars(
            TravisPullRequestBuildDiffer.EXPECTED_TRAVIS_ENV_VARS)
    except UnexpectedTravisEnvironmentError:
        return False
    else:
        return True


class TravisPullRequestBuildDiffer(PullRequestBuildDiffer):
    """Differ from within Travis test environment

    Requires several environment variables set by Travis to work properly:
        - TRAVIS_BUILD_DIR, to detect the git repo
        - TRAVIS_PULL_REQUEST, to determine whether the build is on a pull
            request
        - TRAVIS_COMMIT_RANGE, to determine which commits to diff.

    Args:
        repo: repo object for the project
    """

    EXPECTED_TRAVIS_ENV_VARS = (
        # 'TRAVIS_BRANCH',
        'TRAVIS_BUILD_DIR',
        'TRAVIS_PULL_REQUEST',
        # 'TRAVIS_PULL_REQUEST_BRANCH',
        'TRAVIS_COMMIT_RANGE',
    )

    def __init__(self, repo: git.Repo = None):
        super().__init__(repo)
        if repo is None:
            repo = self._detect_repo()

    def _check_environment(self):
        ensure_expected_travis_env_vars(
            TravisPullRequestBuildDiffer.EXPECTED_TRAVIS_ENV_VARS)

    def _get_diff_endpoints(self):
        commit_range = get_travis_env_or_fail('TRAVIS_COMMIT_RANGE')
        return get_diff_endpoints_from_commit_range(self.repo, commit_range)

    def _detect_repo(self) -> git.Repo:
        build_dir = get_travis_env_or_fail('TRAVIS_BUILD_DIR')
        return git.Repo(build_dir)
