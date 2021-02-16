from unittest.mock import ANY, patch

import funcy
import pytest
from click.testing import CliRunner

import ballet
from ballet.cli import cli as _cli


@pytest.fixture
def cli():
    runner = CliRunner()
    return funcy.partial(runner.invoke, _cli)


def test_cli_version(cli):
    result = cli('--version')
    assert result.exit_code == 0
    assert ballet.__version__ in result.output


@patch('ballet.templating.render_project_template')
def test_quickstart(mock_render, cli):
    result = cli('quickstart')
    mock_render.assert_called_once_with()
    assert 'Generating new ballet project' in result.output


@patch('ballet.update.update_project_template')
def test_update_project_template(mock_update, cli):
    result = cli('update-project-template --push')
    mock_update.assert_called_once_with(
        push=True, project_template_path=None)
    assert 'Updating project template' in result.output


@patch('ballet.templating.start_new_feature')
def test_start_new_feature(mock_start, cli):
    result = cli('start-new-feature')
    mock_start.assert_called_once_with(branching=True)
    assert 'Starting new feature' in result.output


@patch('ballet.project.Project.from_path')
@patch('ballet.validation.main.validate')
def test_validate_all(mock_validate, mock_from_path, cli):
    result = cli('validate -A')
    mock_validate.assert_called_once_with(ANY, True, True, True, True)
    assert 'Validating project' in result.output
