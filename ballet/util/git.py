import pathlib
import re
from typing import Iterable, Iterator, Optional, Tuple

import git
import requests
from funcy import collecting, complement, lfilter, re_find, silent
from github import Github
from github.Repository import Repository
from stacklog import stacklog

from ballet.exc import BalletError
from ballet.util import one_or_raise
from ballet.util.log import logger

FILE_CHANGES_COMMIT_RANGE = '{a}...{b}'
REV_REGEX = r'[a-zA-Z0-9_/^@{}-]+'
COMMIT_RANGE_REGEX = re.compile(
    fr'(?P<a>{REV_REGEX})\.\.(?P<thirddot>\.?)(?P<b>{REV_REGEX})')
GIT_PUSH_FAILURE = (
    git.PushInfo.REJECTED |
    git.PushInfo.REMOTE_REJECTED |
    git.PushInfo.REMOTE_FAILURE |
    git.PushInfo.ERROR
)
DEFAULT_BRANCH = 'master'


class Differ:

    def diff(self) -> git.DiffIndex:
        a, b = self._get_diff_endpoints()
        return a.diff(b)

    def _get_diff_endpoints(self) -> Tuple[git.Diffable, git.Diffable]:
        raise NotImplementedError


class CustomDiffer(Differ):

    def __init__(self, endpoints: Tuple[git.Diffable, git.Diffable]):
        self.endpoints = endpoints

    def _get_diff_endpoints(self) -> Tuple[git.Diffable, git.Diffable]:
        return self.endpoints


class PullRequestBuildDiffer(Differ):
    """Diff files from this pull request against a comparison ref

    Args:
        repo: repo
    """

    def __init__(self, repo: git.Repo):
        self.repo = repo
        self._check_environment()

    def _check_environment(self):
        raise NotImplementedError


class NoOpDiffer(PullRequestBuildDiffer):
    """A differ that returns an empty changeset"""

    def diff(self):
        return []

    def _check_environment(self):
        pass


def can_use_local_differ(repo: Optional[git.Repo]):
    """On some non-master branch"""
    return repo is not None and repo.head.ref.name != 'master'


class LocalPullRequestBuildDiffer(PullRequestBuildDiffer):

    @property
    def _ref_name(self) -> str:
        return self.repo.head.ref.name

    @property
    def _ref_path(self) -> str:
        return self.repo.head.ref.path

    def _check_environment(self):
        assert self._ref_name != 'master'

    def _get_diff_endpoints(self) -> Tuple[git.Diffable, git.Diffable]:
        a = self.repo.rev_parse('master')
        b = self.repo.rev_parse(self._ref_name)
        return a, b


def can_use_local_merge_differ(repo: Optional[git.Repo]):
    """Check the repo HEAD is on master after a merge commit

    Checks for two qualities of the current project:

    1. The project repo's head is the master branch
    2. The project repo's head commit is a merge commit.

    Note that fast-forward style merges will not cause the second condition
    to evaluate to true.
    """
    if repo:
        return (
            get_branch(repo) == 'master' and
            is_merge_commit(repo.head.commit)
        )


class LocalMergeBuildDiffer(Differ):
    """Diff files on a merge commit on the current active branch.

    Merge parent order is guaranteed such that parent 1 is HEAD and parent 2
    is topic [1].

    Attributes:
        repo: The repository to check the merge diff on. Must be currently on
            a branch where the most recent commit is a merge.

    References:
        [1] https://git-scm.com/book/en/v2/Git-Tools-Advanced-Merging
    """

    def __init__(self, repo: git.Repo):
        self.repo = repo
        self._check_environment()

    def _check_environment(self):
        assert is_merge_commit(self.repo.head.commit)

    def _get_diff_endpoints(self) -> Tuple[git.Diffable, git.Diffable]:
        parents = self.repo.head.commit.parents
        return parents[0], parents[1]


def make_commit_range(a: str, b: str) -> str:
    return FILE_CHANGES_COMMIT_RANGE.format(a=a, b=b)


def get_diff_endpoints_from_commit_range(
    repo: git.Repo, commit_range: str
) -> Tuple[git.Diffable, git.Diffable]:
    """Get endpoints of a diff given a commit range

    The resulting endpoints can be diffed directly::

        a, b = get_diff_endpoints_from_commit_range(repo, commit_range)
        a.diff(b)

    For details on specifying git diffs, see ``git diff --help``.
    For details on specifying revisions, see ``git help revisions``.

    Args:
        repo: Repo object initialized with project root
        commit_range: commit range as would be interpreted by ``git
            diff`` command. Unfortunately only patterns of the form ``a..b``
            and ``a...b`` are accepted. Note that the latter pattern finds the
            merge-base of a and b and uses it as the starting point for the
            diff.

    Returns:
        starting commit, ending commit (inclusive)

    Raises:
        ValueError: commit_range is empty or ill-formed

    See also:

        <https://stackoverflow.com/q/7251477>
    """
    if not commit_range:
        raise ValueError('commit_range cannot be empty')

    result = re_find(COMMIT_RANGE_REGEX, commit_range)
    if not result:
        raise ValueError(
            f'Expected diff str of the form \'a..b\' or \'a...b\' '
            f'(got {commit_range})')
    a, b = result['a'], result['b']
    a, b = repo.rev_parse(a), repo.rev_parse(b)
    if result['thirddot']:
        a = one_or_raise(repo.merge_base(a, b))
    return a, b


def get_repo(repo: Optional[git.Repo] = None) -> git.Repo:
    if repo is None:
        repo = git.Repo(pathlib.Path.cwd(),
                        search_parent_directories=True)
    return repo


@silent
def get_branch(repo: Optional[git.Repo] = None) -> str:
    repo = get_repo(repo)
    branch = repo.head.ref.name
    return branch


def switch_to_new_branch(repo: git.Repo, name: str):
    new_branch = repo.create_head(name)
    repo.head.ref = new_branch


def is_merge_commit(commit: git.Commit) -> bool:
    return len(commit.parents) > 1


def set_config_variables(repo: git.Repo, variables: dict):
    """Set config variables

    Args:
        repo: repo
        variables: entries of the form ``'user.email': 'you@example.com'``
    """
    with repo.config_writer() as writer:
        for k, value in variables.items():
            section, option = k.split('.')
            writer.set_value(section, option, value)
        writer.release()


def get_pull_requests(owner: str, repo: str, state: str = 'closed') -> dict:
    base = 'https://api.github.com'
    q = f'/repos/{owner}/{repo}/pulls'
    url = base + q
    headers = {
        'Accept': 'application/vnd.github.v3+json'
    }
    params = {
        'state': state,
        'base': 'master',
        'sort': 'created',
        'direction': 'asc',
    }
    res = requests.get(url, headers=headers, params=params)
    res.raise_for_status()
    return res.json()


@collecting
def get_pull_request_outcomes(owner: str, repo: str) -> Iterator[str]:
    prs = get_pull_requests(owner, repo, state='closed')
    for pr in prs:
        if pr['merged_at'] is not None:
            yield 'accepted'
        else:
            yield 'rejected'


def did_git_push_succeed(push_info: git.remote.PushInfo) -> bool:
    """Check whether a git push succeeded

    A git push succeeded if it was not "rejected" or "remote rejected",
    and if there was not a "remote failure" or an "error".

    Args:
        push_info: push info
    """
    return push_info.flags & GIT_PUSH_FAILURE == 0


def create_github_repo(github: Github, owner: str, name: str) -> Repository:
    """Create the repo ``:owner/:name``

    The authenticated account must have the permissions to create the desired
    repo.

    1. if the desired owner is the user, then this is straightforward
    2. if the desired owner is an organization, then the user must have
       permission to create a new repo for the organization

    Returns:
        the created repository

    Raises:
        github.GithubException.BadCredentialsException: if the token does not
            have permission to create the desired repo
    """
    user = github.get_user()
    if user.login == owner:
        return user.create_repo(name)
    else:
        return github.get_organization(owner).create_repo(name)


@stacklog(logger.info, 'Pushing branches to remote')
def push_branches_to_remote(
    repo: git.Repo,
    remote_name: str,
    branches: Iterable[str]
):
    """Push selected branches to origin

    Similar to::

        $ git push origin branch1:branch1 branch2:branch2

    Raises:
        ballet.exc.BalletError: Push failed in some way
    """
    remote = repo.remote(remote_name)
    result = remote.push([
        f'{b}:{b}'
        for b in branches
    ])
    failures = lfilter(complement(did_git_push_succeed), result)
    if failures:
        for push_info in failures:
            logger.error(
                f'Failed to push ref {push_info.local_ref.name} to '
                f'{push_info.remote_ref.name}')
        raise BalletError('Push failed')
