import pathlib
from unittest.mock import patch

from ballet.templating import (
    _stringify_path, render_feature_template, render_project_template,)


def test__stringify_path_pathlike():
    obj = pathlib.Path()
    output = _stringify_path(obj)
    assert isinstance(output, str)


def test__stringify_path_not_pathlike():
    obj = object()
    output = _stringify_path(obj)
    assert not isinstance(output, str)


@patch('ballet.templating.cookiecutter')
def test_render_project_template(mock_cookiecutter):
    render_project_template()

    args, _ = mock_cookiecutter.call_args

    assert len(args) == 1
    path = args[0]
    assert 'project_template' in str(path)


@patch('ballet.templating.cookiecutter')
def test_render_feature_template(mock_cookiecutter):
    render_feature_template()

    args, _ = mock_cookiecutter.call_args

    assert len(args) == 1
    path = args[0]
    assert 'feature_template' in str(path)
