import importlib
from textwrap import dedent
from types import ModuleType
from unittest.mock import create_autospec, patch

import git
import pytest

from ballet.project import Project
from ballet.util.git import CustomDiffer, Differ
from ballet.validation.common import ChangeCollector, NewFeatureInfo
from ballet.validation.feature_api.validator import FeatureApiValidator
from ballet.validation.project_structure.validator import (
    ProjectStructureValidator,)

from ..util import make_mock_commit, make_mock_commits


@pytest.fixture
def invalid_feature_code():
    return dedent(
        '''
        from sklearn.base import BaseEstimator, TransformerMixin
        class RaisingTransformer(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None, **fit_kwargs):
                raise RuntimeError
            def transform(self, X, **transform_kwargs):
                raise RuntimeError
        input = 'size'
        transformer = RaisingTransformer()
        '''
    ).strip()


@pytest.fixture
def import_error_code():
    return dedent(
        '''
        edf foo():
            pass
        '''
    ).strip()


def code_to_module(code: str, modname='modname') -> ModuleType:
    # see https://stackoverflow.com/a/53080237
    spec = importlib.util.spec_from_loader(modname, loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(code, module.__dict__)
    return module


def test_change_collector_init():
    project = create_autospec(Project)
    differ = Differ()
    change_collector = ChangeCollector(project, differ=differ)
    assert change_collector is not None


@patch('ballet.validation.common.can_use_travis_differ', return_value=True)
@patch('ballet.validation.common.TravisPullRequestBuildDiffer')
def test_change_collector_detect_differ_travis(mock_travis_differ, _):
    """Check ChangeCollector._detect_differ"""
    differ_instance = mock_travis_differ.return_value
    project = create_autospec(Project)
    change_collector = ChangeCollector(project)
    assert change_collector.differ is differ_instance


def test_change_collector_collect_file_diffs_custom_differ(mock_repo):
    repo = mock_repo

    n = 10
    filename = 'file{i}.py'

    commits = make_mock_commits(repo, n=n, filename=filename)

    project = None
    differ = CustomDiffer(endpoints=(commits[0], commits[-1]))
    change_collector = ChangeCollector(project, differ=differ)
    file_diffs = change_collector._collect_file_diffs()

    # checks on file_diffs
    assert len(file_diffs) == n - 1

    for diff in file_diffs:
        assert diff.change_type == 'A'
        assert diff.b_path.startswith('file')
        assert diff.b_path.endswith('.py')


@pytest.mark.xfail
def test_change_collector_categorize_file_diffs():
    raise NotImplementedError


@pytest.mark.xfail
def test_change_collector_collect_features():
    raise NotImplementedError


def test_change_collector_collect_changes(quickstart):
    repo = quickstart.repo
    contrib_path = quickstart.project.config.get('contrib.module_path')

    path_content = [
        ('something.txt', None),  # invalid
        ('invalid.py', None),  # invalid
        (f'{contrib_path}/foo/bar/baz.py', None),  # invalid

        # candidate_feature, and also new_feature_info
        (f'{contrib_path}/user_foo/feature_bar.py', None),

        (f'{contrib_path}/user_foo/__init__.py', None),  # valid_init
    ]

    old_head = repo.head.commit

    for path, content in path_content:
        make_mock_commit(repo, path=path, content=content)

    new_head = repo.head.commit

    differ = CustomDiffer(endpoints=(old_head, new_head))
    change_collector = ChangeCollector(quickstart.project, differ=differ)
    changes = change_collector.collect_changes()

    assert len(changes.file_diffs) == 5
    assert len(changes.candidate_feature_diffs) == 1
    assert len(changes.valid_init_diffs) == 1
    assert len(changes.inadmissible_diffs) == 3
    assert len(changes.new_feature_info) == 1

    actual_inadmissible = [
        diff.b_path
        for diff in changes.inadmissible_diffs
    ]
    expected_inadmissible = [
        'something.txt', 'invalid.py', f'{contrib_path}/foo/bar/baz.py'
    ]
    assert set(actual_inadmissible) == set(expected_inadmissible)


@pytest.mark.parametrize(
    'inadmissible_diffs,expected',
    [
        ([create_autospec(git.Diff)], False),
        ([], True),
    ]
)
@patch('ballet.validation.project_structure.validator.ChangeCollector')
def test_project_structure_validator(
    mock_change_collector, inadmissible_diffs, expected,
):
    mock_change_collector \
        .return_value \
        .collect_changes \
        .return_value \
        .inadmissible_diffs = inadmissible_diffs

    project = None
    validator = ProjectStructureValidator(project)
    result = validator.validate()
    assert result == expected


@patch('ballet.validation.feature_api.validator.ChangeCollector')
def test_feature_api_validator_validation_failure_no_features_found(
    mock_change_collector,
    sample_data,
):
    """
    If the change collector does not return any new features, validation fails.
    """
    mock_change_collector \
        .return_value \
        .collect_changes \
        .return_value \
        .new_feature_info = []

    project = create_autospec(Project)
    project.api.load_data.return_value = sample_data.X, sample_data.y
    validator = FeatureApiValidator(project)
    result = validator.validate()
    assert not result


@patch(
    'ballet.validation.feature_api.validator.validate_feature_api',
    return_value=True
)
@patch('ballet.validation.feature_api.validator.ChangeCollector')
def test_feature_api_validator_validation_failure_invalid_feature(
    mock_change_collector, mock_validate_feature_api,
    sample_data, invalid_feature_code,
):
    """
    If change collector returns a feature, but validate_feature_api fails,
    then validation fails.
    """
    mock_change_collector \
        .return_value \
        .collect_changes \
        .return_value \
        .new_feature_info = [
            NewFeatureInfo(
                lambda: code_to_module(invalid_feature_code),
                'modname',
                'modpath',
            ),
        ]

    project = create_autospec(Project)
    project.api.load_data.return_value = sample_data.X, sample_data.y
    validator = FeatureApiValidator(project)
    result = validator.validate()
    assert not result


@patch('ballet.validation.feature_api.validator.ChangeCollector')
def test_feature_api_validator_validation_failure_import_error(
    mock_change_collector,
    sample_data, import_error_code,
):
    """
    If change collector returns a feature, but importer() fails to return a
    module, then validation fails.
    """
    mock_change_collector \
        .return_value \
        .collect_changes \
        .return_value \
        .new_feature_info = [
            NewFeatureInfo(
                lambda: code_to_module(import_error_code),
                'modname',
                'modpath',
            ),
        ]

    project = create_autospec(Project)
    project.api.load_data.return_value = sample_data.X, sample_data.y
    validator = FeatureApiValidator(project)
    result = validator.validate()
    assert not result
