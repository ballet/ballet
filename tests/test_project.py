import pathlib
import random
import shlex
import subprocess
import sys
from unittest.mock import ANY, PropertyMock, patch

import git
import pytest

from ballet.project import Project, detect_github_username, load_config
from ballet.util.testing import seeded


@patch('ballet.project.load_config_in_dir')
def test_load_config(mock_load_config_in_dir):
    path = pathlib.Path(__file__)
    load_config(path=path)
    mock_load_config_in_dir.assert_called_once_with(path)


@patch('ballet.project.load_config_in_dir')
def test_load_config_detect(mock_load_config_in_dir):
    path = pathlib.Path(__file__)
    load_config()
    mock_load_config_in_dir.assert_called_once_with(path)


@pytest.mark.skipif(
    sys.version_info < (3, 7),
    reason='capture_output added in py37, not worth it to add compat')
@patch('ballet.project.load_config_in_dir')
def test_load_config_dash_c(mock_load_config_in_dir, quickstart):
    # note that this tests python -c 'cmd' which is different from the repl!
    pycmd = 'from ballet.project import load_config; config=load_config()'
    cmd = f"{sys.executable} -c '{pycmd}'"
    result = subprocess.run(
        shlex.split(cmd), cwd=quickstart.project.path, capture_output=True)
    assert not result.returncode, result.stderr


@pytest.mark.xfail
def test_load_config_at_path():
    raise NotImplementedError


@pytest.mark.xfail
def test_load_config_in_dir():
    raise NotImplementedError


@patch('ballet.project.Project.repo', new_callable=PropertyMock)
def test_detect_github_username_config(mock_project_repo):
    expected_username = 'Foo Bar'

    # output of project.repo.config_reader().get_value(...)
    mock_get_value = (mock_project_repo
                      .return_value
                      .config_reader
                      .return_value
                      .get_value)
    mock_get_value.return_value = expected_username

    project = Project(None)
    username = detect_github_username(project)
    assert expected_username == username

    mock_get_value.assert_called_with('github', 'user', default=ANY)


@pytest.fixture
def commit_object():
    with seeded(17):
        # 20-byte sha1
        k = 20
        bits = random.getrandbits(k * 8)
        data = bits.to_bytes(k, sys.byteorder)  # in py39, can use randbytes(k)

        repo = None
        commit = git.Commit(repo, data)
        yield commit


@patch('ballet.project.Project.repo', new_callable=PropertyMock)
def test_project_version(mock_repo, commit_object):
    mock_repo.return_value.head.commit = commit_object
    project = Project(None)
    version = project.version
    assert isinstance(version, str)
