import os
import re

import git
import requests
from funcy import collecting, ignore, re_find, re_test

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
        diff_str = self._get_diff_str()
        diffs = get_diffs_by_diff_str(self.repo, diff_str)
        return diffs

    def _check_environment(self):
        raise NotImplementedError

    def _get_diff_str(self):
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

    def _get_diff_str(self):
        return '{from_}..{to_}'.format(from_='master', to_=self._pr_name)


def get_diffs_by_revision(repo, from_revision, to_revision):
    '''Get file changes between two revisions.

    For details on specifying revisions, see `git help revisions`.

    Args:
        repo (git.Repo): Repo object initialized with project root
        from_revision (str): revision identifier for the starting point of the
            diff
        to_revision (str): revision identifier for the ending point of the diff

    Returns:
        list of git.diff.Diff identifying changes between revisions
    '''
    diff_str = '{from_revision}..{to_revision}'.format(
        from_revision=from_revision, to_revision=to_revision)
    return get_diffs_by_diff_str(repo, diff_str)


def get_diff_str_from_commits(a, b):
    return '{a}..{b}'.format(a=a.hexsha, b=b.hexsha)


def get_diffs_by_diff_str(repo, diff_str):
    '''Get file changes via a diff string.

    For details on specifying revisions, see `git help revisions`.

    Args:
        repo (git.Repo): Repo object initialized with project root
        diff_str (str): diff string identifying range of diff. For example,
            `master..HEAD` diffs from master to HEAD, and `12345678..abcdef90`
            compares to commits.

    Returns:
        list of git.diff.Diff identifying changes between revisions
    '''
    a, b = diff_str.split('..')
    a_obj = repo.rev_parse(a)
    b_obj = repo.rev_parse(b)
    diffs = a_obj.diff(b_obj)
    return diffs


@ignore(Exception)
def get_pr_num(repo=None):
    if repo is None:
        repo = git.Repo(os.getcwd(), search_parent_directories=True)
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
