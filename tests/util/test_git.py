from unittest.mock import create_autospec, patch

import git
import pytest
from github import BadCredentialsException, Github

from ballet.util.git import (
    create_github_repo, did_git_push_succeed, get_pull_request_outcomes,
    get_pull_requests, make_commit_range,)


def test_make_commit_range():
    a = 'abc1234'
    b = 'def4321'
    expected_commit_range = 'abc1234...def4321'
    actual_commit_range = make_commit_range(a, b)
    assert actual_commit_range == expected_commit_range


@pytest.mark.xfail
def test_get_diff_endpoints_from_commit_range():
    raise NotImplementedError


@pytest.mark.xfail
def test_get_repo():
    raise NotImplementedError


@pytest.mark.xfail
def test_get_branch():
    raise NotImplementedError


@pytest.mark.xfail
def test_switch_to_new_branch():
    raise NotImplementedError


@pytest.mark.xfail
def test_set_config_variables():
    raise NotImplementedError


@patch('requests.get')
def test_get_pull_requests(mock_requests_get):
    owner = 'foo'
    repo = 'bar'
    state = 'closed'
    get_pull_requests(owner, repo, state=state)

    (url, ), kwargs = mock_requests_get.call_args
    assert owner in url
    assert repo in url
    assert 'headers' in kwargs
    assert 'params' in kwargs
    assert kwargs['params']['state'] == state


@patch('ballet.util.git.get_pull_requests')
def test_get_pull_request_outcomes(mock_get_pull_requests):
    mock_get_pull_requests.return_value = [
        {
            'id': 1,
            "created_at": "2011-01-26T19:01:12Z",
            "updated_at": "2011-01-26T19:01:12Z",
            "closed_at": "2011-01-26T19:01:12Z",
            "merged_at": "2011-01-26T19:01:12Z",
        },
        {
            "created_at": "2011-01-26T19:03:19Z",
            "updated_at": "2011-01-26T19:03:19Z",
            "closed_at": "2011-01-26T19:04:01Z",
            "merged_at": None,
        }
    ]
    owner = 'foo'
    repo = 'bar'

    expected = ['accepted', 'rejected']
    actual = get_pull_request_outcomes(owner, repo)
    assert actual == expected
    mock_get_pull_requests.assert_called_once_with(
        owner, repo, state='closed')


def test_did_git_push_succeed():
    local_ref = None
    remote_ref_string = None
    remote = None

    flags = 0
    push_info = git.remote.PushInfo(flags, local_ref, remote_ref_string,
                                    remote)
    assert did_git_push_succeed(push_info)

    flags = git.remote.PushInfo.ERROR
    push_info = git.remote.PushInfo(flags, local_ref, remote_ref_string,
                                    remote)
    assert not did_git_push_succeed(push_info)


@pytest.fixture
def github():
    return create_autospec(Github)


@pytest.mark.parametrize(
    'owner',
    ['octocat', 'github-dot-com'],
)
def test_create_github_repo(github, owner):
    user = 'octocat'  # noqa
    repo = 'Hello-World'
    repository = create_github_repo(github, owner, repo)
    assert repository.full_name == f'{owner}/{repo}'


@pytest.mark.parametrize(
    'owner',
    ['octocat', 'github-dot-com'],
)
def test_create_github_repo_not_authorized(github, owner):
    user = 'octocat'  # noqa
    repo = 'Hello-World'
    with pytest.raises(BadCredentialsException):
        create_github_repo(github, owner, repo)
