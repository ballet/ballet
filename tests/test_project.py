from unittest.mock import ANY, PropertyMock, patch

import pytest

from ballet.project import Project, detect_github_username


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


@patch('ballet.project.Project.repo', new_callable=PropertyMock)
@patch('ballet.project.get_pr_num')
def test_project_pr_num(mock_get_pr_num, mock_repo):
    expected = 3
    mock_get_pr_num.return_value = expected

    package = None
    project = Project(package)
    assert project.pr_num == expected
