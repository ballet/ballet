from unittest.mock import patch

import pytest

from ballet.util.ci import (
    TravisPullRequestBuildDiffer, get_travis_branch, get_travis_pr_num,
    is_travis_pr,)
from ballet.util.git import make_commit_range
from tests.util import make_mock_commit, make_mock_commits


@pytest.fixture
def pr_num():
    return 7


@pytest.fixture
def commit_range():
    return 'HEAD^..HEAD'


def test_get_travis_pr_num(pr_num):
    # matrix of env name, setting for env, expected result
    matrix = (
        ('TRAVIS_PULL_REQUEST', str(pr_num), pr_num),
        ('TRAVIS_PULL_REQUEST', 'true', None),
        ('TRAVIS_PULL_REQUEST', 'FALSE', None),
        ('TRAVIS_PULL_REQUEST', 'false', None),
        ('TRAVIS_PULL_REQUEST', 'abcd', None),
        ('UNRELATED', '', None),
    )
    for env_name, env_value, expected in matrix:
        with patch.dict('os.environ', {env_name: env_value}, clear=True):
            actual = get_travis_pr_num()
            assert actual == expected


def test_is_travis_pr(pr_num):
    matrix = (
        ('TRAVIS_PULL_REQUEST', str(pr_num), True),
        ('TRAVIS_PULL_REQUEST', 'true', False),
        ('TRAVIS_PULL_REQUEST', 'FALSE', False),
        ('TRAVIS_PULL_REQUEST', 'false', False),
        ('TRAVIS_PULL_REQUEST', 'abcd', False),
        ('UNRELATED', '', False),
    )
    for env_name, env_value, expected in matrix:
        with patch.dict('os.environ', {env_name: env_value}, clear=True):
            actual = is_travis_pr()
            assert actual == expected


def test_get_travis_branch():
    # matrix of env dict, expected result
    matrix = (
        ({
            'TRAVIS_PULL_REQUEST': 'false',
            'TRAVIS_PULL_REQUEST_BRANCH': '',
            'TRAVIS_BRANCH': 'master',
        }, 'master'),
        ({
            'TRAVIS_PULL_REQUEST': 'false',
            'TRAVIS_PULL_REQUEST_BRANCH': '',
            'TRAVIS_BRANCH': 'foo',
        }, 'foo'),
        ({
            'TRAVIS_PULL_REQUEST': '1',
            'TRAVIS_PULL_REQUEST_BRANCH': 'foo',
            'TRAVIS_BRANCH': 'master',
        }, 'foo'),
        ({}, None),
    )

    for env, expected in matrix:
        with patch.dict('os.environ', env, clear=True):
            actual = get_travis_branch()
            assert actual == expected


def test_travis_pull_request_build_differ(mock_repo, pr_num, commit_range):
    repo = mock_repo
    make_mock_commits(repo, n=3)

    travis_env_vars = {
        'TRAVIS_BUILD_DIR': repo.working_tree_dir,
        'TRAVIS_PULL_REQUEST': str(pr_num),
        'TRAVIS_COMMIT_RANGE': commit_range,
    }
    with patch.dict('os.environ', travis_env_vars, clear=True):
        differ = TravisPullRequestBuildDiffer(pr_num)
        expected_a = repo.rev_parse('HEAD^')
        expected_b = repo.rev_parse('HEAD')
        actual_a, actual_b = differ._get_diff_endpoints()
        assert actual_a == expected_a
        assert actual_b == expected_b


def test_travis_pull_request_build_differ_on_mock_commits(mock_repo, pr_num):
    repo = mock_repo
    n = 4
    i = 0
    feature_branch_name = 'pull/{}'.format(pr_num)

    make_mock_commit(repo, path='readme.txt')
    expected_merge_base = repo.head.commit
    feature_branch = repo.create_head(feature_branch_name)

    # make commits on branch master
    commits = make_mock_commits(repo, n=3, filename='blah{i}.txt')
    master = repo.heads.master

    # make commits on feature branch
    feature_branch.checkout()
    commits = make_mock_commits(repo, n=n)
    end_commit = commits[-1]

    commit_range = make_commit_range(
        master, end_commit)

    travis_env_vars = {
        'TRAVIS_BUILD_DIR': repo.working_tree_dir,
        'TRAVIS_PULL_REQUEST': str(pr_num),
        'TRAVIS_COMMIT_RANGE': commit_range,
    }
    with patch.dict('os.environ', travis_env_vars, clear=True):
        differ = TravisPullRequestBuildDiffer(pr_num)
        a, b = differ._get_diff_endpoints()
        assert a == expected_merge_base
        assert b == end_commit

        diffs = differ.diff()

        # there should be n diff objects, they should show files
        # 0 to n-1. merge base just created readme.txt, so all
        # files on feature branch are new.
        assert len(diffs) == n
        j = i
        for diff in diffs:
            assert diff.change_type == 'A'
            assert diff.b_path == 'file{j}.py'.format(j=j)
            j += 1
