import re

import git
import requests
from funcy import collecting, ignore, re_find, re_test

from ballet.compat import pathlib, safepath
from ballet.util import one_or_raise

FILE_CHANGES_COMMIT_RANGE = '{a}...{b}'
REV_REGEX = r'[a-zA-Z0-9_/^@{}-]+'
COMMIT_RANGE_REGEX = re.compile(
    r'(?P<a>{rev})\.\.(?P<thirddot>\.?)(?P<b>{rev})'
    .format(rev=REV_REGEX))
PR_REF_PATH_REGEX = re.compile(r'refs/heads/pull/(\d+)')


class PullRequestBuildDiffer:
    """Diff files from this pull request against a comparison ref

    Args:
        pr_num (str, int): pull request number
        repo (git.Repo): repo
    """

    def __init__(self, pr_num, repo):
        self.pr_num = int(pr_num)
        self.repo = repo
        self._check_environment()

    def diff(self):
        a, b = self._get_diff_endpoints()
        return a.diff(b)

    def _check_environment(self):
        raise NotImplementedError

    def _get_diff_endpoints(self):
        raise NotImplementedError


class LocalPullRequestBuildDiffer(PullRequestBuildDiffer):

    @property
    def _pr_name(self):
        return self.repo.head.ref.name

    @property
    def _pr_path(self):
        return self.repo.head.ref.path

    def _check_environment(self):
        assert re_test(PR_REF_PATH_REGEX, self._pr_path)

    def _get_diff_endpoints(self):
        a = self.repo.rev_parse('master')
        b = self.repo.rev_parse(self._pr_name)
        return a, b


def make_commit_range(a, b):
    return FILE_CHANGES_COMMIT_RANGE.format(a=a, b=b)


def get_diff_endpoints_from_commit_range(repo, commit_range):
    """Get file changes via a commit range

    For details on specifying revisions, see `git help revisions`.

    Args:
        repo (git.Repo): Repo object initialized with project root
        diff_str (str): diff string identifying range of diff as would be
            interpreted by ``git diff`` command. Unfortunately only patterns of
            the form ``a..b`` and ``a...b`` are accepted. For more details on
            the difference between these two forms,
            see https://stackoverflow.com/q/7251477.

    Returns:
        List[git.diff.Diff]: changes between revisions
    """
    if not commit_range:
        raise ValueError('commit_range cannot be empty')

    result = re_find(COMMIT_RANGE_REGEX, commit_range)
    if not result:
        raise ValueError(
            'Expected diff str of the form \'a..b\' or \'a...b\' (got {})'
            .format(commit_range))
    a, b = result['a'], result['b']
    a, b = repo.rev_parse(a), repo.rev_parse(b)
    if result['thirddot']:
        a = one_or_raise(repo.merge_base(a, b))
    return a, b


@ignore(Exception)
def get_pr_num(repo=None):
    if repo is None:
        repo = git.Repo(safepath(pathlib.Path.cwd()),
                        search_parent_directories=True)
    pr_num = re_find(PR_REF_PATH_REGEX, repo.head.ref.path)
    return int(pr_num)


def switch_to_new_branch(repo, name):
    new_branch = repo.create_head(name)
    repo.head.ref = new_branch


def set_config_variables(repo, variables):
    """Set config variables

    Args:
        variables (dict): entries of the form 'user.email': 'you@example.com'
    """
    for k, v in variables.items():
        repo.git.config(k, v)


def get_pull_requests(owner, repo, state='closed'):
    base = 'https://api.github.com'
    q = '/repos/{owner}/{repo}/pulls'.format(owner=owner, repo=repo)
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
def get_pull_request_outcomes(owner, repo):
    prs = get_pull_requests(owner, repo, state='closed')
    for pr in prs:
        if pr['merged_at'] is not None:
            yield 'accepted'
        else:
            yield 'rejected'
