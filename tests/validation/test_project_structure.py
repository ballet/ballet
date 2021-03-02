from textwrap import dedent
from unittest.mock import create_autospec, patch

import pytest

from ballet.util.ci import TravisPullRequestBuildDiffer
from ballet.util.git import CustomDiffer
from ballet.validation.common import ChangeCollector
from ballet.validation.project_structure.validator import (
    ProjectStructureValidator,)

from ..util import make_mock_commit, make_mock_commits
from .conftest import mock_feature_api_validator


@pytest.fixture
def pr_num():
    return 73


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


def test_change_collector_init(null_change_collector):
    assert isinstance(
        null_change_collector.differ, TravisPullRequestBuildDiffer)


def test_change_collector_collect_file_diffs_custom_differ(pr_num, mock_repo):
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


def test_change_collector_collect_changes(
    pr_num, quickstart,
):
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
        ([create_autospec('git.Diff')], False),
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


def test_feature_api_validator_validation_failure_no_features_found(
    pr_num, sample_data
):
    path_content = [
        ('readme.txt', None),
        ('src/foo/__init__.py', None),
        ('src/foo/contrib/__init__.py', None),
        ('src/foo/contrib/user_foo/__init__.py', None),
        ('src/foo/contrib/user_foo/feature_bar.py', None),
    ]
    contrib_module_path = 'src/foo/contrib/'
    with mock_feature_api_validator(
        path_content, pr_num, contrib_module_path, sample_data.X, sample_data.y
    ) as validator:
        result = validator.validate()
        assert not result


def test_feature_api_validator_validation_failure_invalid_feature(
    sample_data, pr_num, invalid_feature_code,
):
    path_content = [
        ('foo.jpg', None),
        ('src/foo/__init__.py', None),
        ('src/foo/contrib/__init__.py', None),
        ('src/foo/contrib/user_foo/__init__.py', None),
        ('src/foo/contrib/user_foo/feature_bar.py',
            invalid_feature_code),
    ]
    contrib_module_path = 'src/foo/contrib/'
    with mock_feature_api_validator(
        path_content, pr_num, contrib_module_path,
        sample_data.X, sample_data.y
    ) as validator:
        changes = validator.change_collector.collect_changes()
        assert len(changes.file_diffs) == 1
        assert len(changes.candidate_feature_diffs) == 1
        assert len(changes.valid_init_diffs) == 0
        assert len(changes.inadmissible_diffs) == 0

        # TODO
        # self.assertEqual(len(new_features), 1)
        # self.assertTrue(imported_okay)

        result = validator.validate()
        assert not result


def test_feature_api_validator_validation_failure_import_error(
    sample_data, pr_num
):
    import_error_str = dedent('''
        edf foo():  pass
    ''').strip()
    path_content = [
        ('foo.jpg', None),
        ('src/foo/__init__.py', None),
        ('src/foo/contrib/__init__.py', None),
        ('src/foo/contrib/user_foo/__init__.py', None),
        ('src/foo/contrib/user_foo/feature_baz.py', import_error_str),
    ]
    contrib_module_path = 'src/foo/contrib/'
    with mock_feature_api_validator(
        path_content, pr_num, contrib_module_path, sample_data.X, sample_data.y
    ) as validator:
        changes = validator.change_collector.collect_changes()
        assert len(changes.file_diffs) == 1
        assert len(changes.candidate_feature_diffs) == 1
        assert len(changes.valid_init_diffs) == 0
        assert len(changes.inadmissible_diffs) == 0

        # TODO
        # self.assertEqual(len(new_feature_info), 0)
        # self.assertFalse(imported_okay)

        result = validator.validate()
        assert not result
