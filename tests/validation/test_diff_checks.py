import pathlib
from unittest.mock import Mock

import pytest

from ballet.project import relative_to_contrib
from ballet.validation.project_structure.checks import (
    IsAdditionCheck, IsPythonSourceCheck, ModuleNameCheck,
    RelativeNameDepthCheck, SubpackageNameCheck, WithinContribCheck,)


# @pytest.fixture(scope='module')  # TODO
@pytest.fixture
def project(quickstart):
    yield quickstart.project


def test_relative_to_contrib(project):
    contrib_path = project.config.get('contrib.module_path')
    diff = Mock(b_path=f'{contrib_path}/abc.py')

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
    contrib_path = project.config.get('contrib.module_path')
    checker = IsPythonSourceCheck(project)

    mock_diff = Mock(b_path=f'{contrib_path}/user_bob/feature_1.py')
    assert checker.do_check(mock_diff)

    mock_diff = Mock(b_path=f'{contrib_path}/user_bob/feature_1.xyz')
    assert not checker.do_check(mock_diff)


def test_within_contrib_check(project):
    contrib_path = project.config.get('contrib.module_path')
    checker = WithinContribCheck(project)

    mock_diff = Mock(b_path=f'{contrib_path}/user_bob/feature_1.py')
    assert checker.do_check(mock_diff)

    mock_diff = Mock(b_path='foo/hack.py')
    assert not checker.do_check(mock_diff)


def test_subpackage_name_check(project):
    contrib_path = project.config.get('contrib.module_path')
    checker = SubpackageNameCheck(project)

    mock_diff = Mock(b_path=f'{contrib_path}/user_bob/feature_1.py')
    assert checker.do_check(mock_diff)

    mock_diff = Mock(b_path=f'{contrib_path}/bob/feature_1.py')
    assert not checker.do_check(mock_diff)


def test_feature_module_name_check(project):
    contrib_path = project.config.get('contrib.module_path')
    checker = ModuleNameCheck(project)

    mock_diff = Mock(b_path=f'{contrib_path}/user_bob/feature_1.py')
    assert checker.do_check(mock_diff)

    bad_paths = [
        f'{contrib_path}/user_bob/foo1.py',
        f'{contrib_path}/user_bob/1.py',
        f'{contrib_path}/user_bob/feature_x-1.py',
    ]
    for path in bad_paths:
        mock_diff = Mock(b_path=path)
        assert not checker.do_check(mock_diff)


def test_relative_name_depth_check(project):
    contrib_path = project.config.get('contrib.module_path')
    checker = RelativeNameDepthCheck(project)

    mock_diff = Mock(b_path=f'{contrib_path}/user_bob/feature_1.py')
    assert checker.do_check(mock_diff)

    mock_diff = Mock(
        b_path=f'{contrib_path}/user_bob/a/b/c/d/feature_1.py')
    assert not checker.do_check(mock_diff)
