from textwrap import dedent
from unittest.mock import patch

import pytest

from ballet.util.ci import TravisPullRequestBuildDiffer
from ballet.util.git import CustomDiffer, make_commit_range
from ballet.validation.common import ChangeCollector

from ..util import make_mock_commits
from .conftest import (
    make_mock_project, mock_feature_api_validator, mock_file_change_validator,)


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


def test_change_collector_collect_file_diffs(pr_num, mock_repo):
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


@pytest.mark.xfail
def test_change_collector_collect_changes():
    raise NotImplementedError


def test_file_change_validator_validation_failure_inadmissible_file_diffs(
    pr_num
):
    path_content = [
        ('readme.txt', None),
        ('src/foo/__init__.py', None),
        ('src/foo/contrib/__init__.py', None),
        ('src/foo/contrib/user_foo/feature_bar.py', None),
        ('src/foo/contrib/user_foo/__init__.py', None),
        ('invalid.py', None),
    ]
    contrib_module_path = 'src/foo/contrib/'
    with mock_file_change_validator(
        path_content, pr_num, contrib_module_path
    ) as validator:
        changes = validator.change_collector.collect_changes()
        assert len(changes.file_diffs) == 1
        assert len(changes.candidate_feature_diffs) == 0
        assert len(changes.valid_init_diffs) == 0
        assert len(changes.inadmissible_diffs) == 1
        assert changes.inadmissible_diffs[0].b_path == 'invalid.py'

        # TODO
        # self.assertTrue(imported_okay)

        result = validator.validate()
        assert not result


def test_file_change_validator_validation_success(pr_num):
    path_content = [
        ('bob.xml', '<hello>'),
        ('src/foo/__init__.py', None),
        ('src/foo/contrib/__init__.py', None),
        ('src/foo/contrib/user_foo/__init__.py', None),
        ('src/foo/contrib/user_foo/feature_bar.py', None),
    ]
    contrib_module_path = 'src/foo/contrib/'
    with mock_file_change_validator(
        path_content, pr_num, contrib_module_path
    ) as validator:
        result = validator.validate()
        assert result


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
