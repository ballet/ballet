import pathlib
from unittest.mock import Mock

import pytest

from ballet.project import relative_to_contrib
from ballet.validation.project_structure.checks import (
    IsAdditionCheck, IsPythonSourceCheck, ModuleNameCheck,
    RelativeNameDepthCheck, SubpackageNameCheck, WithinContribCheck,)

from .util import make_mock_project


@pytest.fixture
def project():
    contrib_module_path = 'foo/features/contrib'
    project = make_mock_project(
        None, None, '', contrib_module_path)
    yield project


def test_relative_to_contrib(project):
    diff = Mock(b_path='foo/features/contrib/abc.py')

    expected = pathlib.Path('abc.py')
    actual = relative_to_contrib(diff, project)

    assert actual == expected


def test_is_addition_check(project):
    checker = IsAdditionCheck(project)

    mock_diff = Mock(change_type='A')
    assert checker.do_check(mock_diff)

    mock_diff = Mock(change_type='B')
    assert not checker.do_check(mock_diff)


def test_is_python_source_check(project):
    checker = IsPythonSourceCheck(project)

    mock_diff = Mock(b_path='foo/features/contrib/user_bob/feature_1.py')
    assert checker.do_check(mock_diff)

    mock_diff = Mock(b_path='foo/features/contrib/user_bob/feature_1.xyz')
    assert not checker.do_check(mock_diff)


def test_within_contrib_check(project):
    checker = WithinContribCheck(project)

    mock_diff = Mock(b_path='foo/features/contrib/user_bob/feature_1.py')
    assert checker.do_check(mock_diff)

    mock_diff = Mock(b_path='foo/hack.py')
    assert not checker.do_check(mock_diff)


def test_subpackage_name_check(project):
    checker = SubpackageNameCheck(project)

    mock_diff = Mock(b_path='foo/features/contrib/user_bob/feature_1.py')
    assert checker.do_check(mock_diff)

    mock_diff = Mock(b_path='foo/features/contrib/bob/feature_1.py')
    assert not checker.do_check(mock_diff)


def test_feature_module_name_check(project):
    checker = ModuleNameCheck(project)

    mock_diff = Mock(b_path='foo/features/contrib/user_bob/feature_1.py')
    assert checker.do_check(mock_diff)

    bad_paths = [
        'foo/features/contrib/user_bob/foo1.py',
        'foo/features/contrib/user_bob/1.py',
        'foo/features/contrib/user_bob/feature_x-1.py',
    ]
    for path in bad_paths:
        mock_diff = Mock(b_path=path)
        assert not checker.do_check(mock_diff)


def test_relative_name_depth_check(project):
    checker = RelativeNameDepthCheck(project)

    mock_diff = Mock(b_path='foo/features/contrib/user_bob/feature_1.py')
    assert checker.do_check(mock_diff)

    mock_diff = Mock(
        b_path='foo/features/contrib/user_bob/a/b/c/d/feature_1.py')
    assert not checker.do_check(mock_diff)
