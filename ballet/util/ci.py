import os

import git
from funcy import constantly, ignore, post_processing

from ballet.exc import UnexpectedTravisEnvironmentError
from ballet.util.git import PullRequestBuildDiffer
from ballet.util.log import logger


def get_travis_env_or_fail(name):
    if name in os.environ:
        return os.environ[name]
    else:
        # dump_travis_env_vars()
        raise UnexpectedTravisEnvironmentError(
            'Missing TRAVIS environment variable: {name}'
            .format(name=name))


def ensure_expected_travis_env_vars(names):
    if not set(names).issubset(set(os.environ.keys())):
        raise UnexpectedTravisEnvironmentError


def dump_travis_env_vars():
    travis_env_vars = {
        k: v for k, v in os.environ.items()
        if k.startswith('TRAVIS')
    }
    logger.info(repr(travis_env_vars))


def get_travis_pr_num():
    """Return the PR number if the job is a pull request, None otherwise

    Returns:
        int

    See also:
        - <https://docs.travis-ci.com/user/environment-variables/#Default-Environment-Variables>
    """  # noqa
    try:
        travis_pull_request = get_travis_env_or_fail('TRAVIS_PULL_REQUEST')
        if travis_pull_request == 'false':
            return None
        else:
            try:
                return int(travis_pull_request)
            except ValueError:
                return None
    except UnexpectedTravisEnvironmentError:
        return None


def is_travis_pr():
    """Check if the current job is a pull request build"""
    return get_travis_pr_num() is not None


@ignore(UnexpectedTravisEnvironmentError, default=False)
@post_processing(constantly(True))
def can_use_travis_differ():
    """Check if the required travis env vars are set for the travis differ"""
    ensure_expected_travis_env_vars(
        TravisPullRequestBuildDiffer.EXPECTED_TRAVIS_ENV_VARS)


class TravisPullRequestBuildDiffer(PullRequestBuildDiffer):

    EXPECTED_TRAVIS_ENV_VARS = (
        'TRAVIS_BUILD_DIR',
        'TRAVIS_PULL_REQUEST',
        'TRAVIS_COMMIT_RANGE',
    )

    def __init__(self, pr_num, repo=None):
        if repo is None:
            repo = self._detect_repo()
        super().__init__(pr_num, repo)

    def _check_environment(self):
        ensure_expected_travis_env_vars(
            TravisPullRequestBuildDiffer.EXPECTED_TRAVIS_ENV_VARS)

        travis_pr_num = get_travis_pr_num()
        if travis_pr_num != self.pr_num:
            raise UnexpectedTravisEnvironmentError(
                'TRAVIS_PULL_REQUEST {tpr!r} did not match expected {pr!r}'
                .format(tpr=travis_pr_num, pr=self.pr_num))

    def _get_diff_str(self):
        return get_travis_env_or_fail('TRAVIS_COMMIT_RANGE')

    def _detect_repo(self):
        build_dir = get_travis_env_or_fail('TRAVIS_BUILD_DIR')
        return git.Repo(build_dir)
