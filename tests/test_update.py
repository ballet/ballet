import json
import logging
from unittest.mock import Mock, patch

import git

import ballet
import ballet.util.log
from ballet.update import (
    PYPI_PROJECT_JSON_URL, _check_for_updated_ballet,
    _get_latest_ballet_version_string, _get_latest_project_version_string,
    _log_recommended_reinstall, _make_template_branch_merge_commit_message,
    _safe_delete_remote, _warn_of_updated_ballet,)


def test_get_latest_project_version_string(testdatadir, responses):
    with testdatadir.joinpath('sampleproject.json').open('r') as f:
        data = json.load(f)
    url = PYPI_PROJECT_JSON_URL.format(project='sampleproject')
    responses.add(responses.GET, url, json=data)
    expected = '1.2.0'

    actual = _get_latest_project_version_string('sampleproject')

    assert actual == expected


@patch('ballet.update._get_latest_project_version_string')
def test_get_latest_ballet_version_string(mock_latest):
    expected = mock_latest.return_value
    actual = _get_latest_ballet_version_string()
    assert actual == expected


@patch('ballet.update._get_latest_ballet_version_string')
def test_check_for_updated_ballet(mock_latest):
    # obviously this will represent an update from whatever the current
    # version is
    latest = '99999999999.0.0'
    mock_latest.return_value = latest
    expected = latest
    actual = _check_for_updated_ballet()
    assert actual == expected


@patch('ballet.update._get_latest_ballet_version_string')
def test_check_for_updated_ballet_no_updates(mock_latest):
    mock_latest.return_value = ballet.__version__
    expected = None  # no updates available
    actual = _check_for_updated_ballet()
    assert actual == expected


def test_warn_of_updated_ballet(caplog):
    caplog.set_level(logging.DEBUG, ballet.util.log.logger.name)
    latest = 'x.y.z'
    _warn_of_updated_ballet(latest)
    assert latest in caplog.text


def test_make_template_branch_merge_commit_message():
    result = _make_template_branch_merge_commit_message()
    assert ballet.__version__ in result


def test_safe_delete_remote():
    repo = Mock(spec=git.Repo)
    name = 'name'
    _safe_delete_remote(repo, name)
    repo.delete_remote.assert_called_once_with(name)


def test_log_recommended_reinstall(caplog):
    caplog.set_level(logging.DEBUG, ballet.util.log.logger.name)
    _log_recommended_reinstall()
    assert caplog.text
