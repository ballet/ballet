import unittest
from textwrap import dedent
from unittest.mock import patch

from ballet.util.ci import TravisPullRequestBuildDiffer
from ballet.util.git import make_commit_range
from ballet.validation.common import ChangeCollector

from .util import (
    SampleDataMixin, make_mock_project, mock_feature_api_validator,
    mock_file_change_validator, null_change_collector)
from ..util import make_mock_commits, mock_repo


class _CommonSetup(SampleDataMixin):

    def setUp(self):
        super().setUp()
        self.pr_num = 73

        self.valid_feature_str = dedent(
            '''
            from sklearn.base import BaseEstimator, TransformerMixin
            class IdentityTransformer(BaseEstimator, TransformerMixin):
                def fit(self, X, y=None, **fit_kwargs):
                    return self
                def transform(self, X, **transform_kwargs):
                    return X
            input = 'size'
            transformer = IdentityTransformer()
            '''
        ).strip()
        self.invalid_feature_str = dedent(
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


class ChangeCollectorTest(_CommonSetup, unittest.TestCase):

    def test_init(self):
        with null_change_collector(self.pr_num) as change_collector:
            self.assertIsInstance(
                change_collector.differ, TravisPullRequestBuildDiffer)

    def test_collect_file_diffs(self):
        n = 10
        filename = 'file{i}.py'
        with mock_repo() as repo:
            commits = make_mock_commits(repo, n=n, filename=filename)
            contrib_module_path = None
            commit_range = make_commit_range(
                commits[0], commits[-1])

            travis_env_vars = {
                'TRAVIS_BUILD_DIR': repo.working_tree_dir,
                'TRAVIS_PULL_REQUEST': str(self.pr_num),
                'TRAVIS_COMMIT_RANGE': commit_range,
            }

            with patch.dict('os.environ', travis_env_vars, clear=True):
                project_path = repo.working_tree_dir
                project = make_mock_project(repo, self.pr_num, project_path,
                                            contrib_module_path)
                change_collector = ChangeCollector(project)
                file_diffs = change_collector._collect_file_diffs()

                # checks on file_diffs
                self.assertEqual(len(file_diffs), n - 1)

                for diff in file_diffs:
                    self.assertEqual(diff.change_type, 'A')
                    self.assertTrue(diff.b_path.startswith('file'))
                    self.assertTrue(diff.b_path.endswith('.py'))

    @unittest.expectedFailure
    def test_categorize_file_diffs(self):
        raise NotImplementedError

    @unittest.expectedFailure
    def test_collect_features(self):
        raise NotImplementedError

    @unittest.expectedFailure
    def test_collect_changes(self):
        raise NotImplementedError


class FileChangeValidatorTest(_CommonSetup, unittest.TestCase):

    def test_validation_failure_inadmissible_file_diffs(self):
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
            path_content, self.pr_num, contrib_module_path
        ) as validator:
            changes = validator.change_collector.collect_changes()
            self.assertEqual(len(changes.file_diffs), 1)
            self.assertEqual(len(changes.candidate_feature_diffs), 0)
            self.assertEqual(len(changes.valid_init_diffs), 0)
            self.assertEqual(len(changes.inadmissible_diffs), 1)
            self.assertEqual(
                changes.inadmissible_diffs[0].b_path, 'invalid.py')

            # TODO
            # self.assertTrue(imported_okay)

            result = validator.validate()
            self.assertFalse(result)

    def test_validation_success(self):
        path_content = [
            ('bob.xml', '<hello>'),
            ('src/foo/__init__.py', None),
            ('src/foo/contrib/__init__.py', None),
            ('src/foo/contrib/user_foo/__init__.py', None),
            ('src/foo/contrib/user_foo/feature_bar.py', None),
        ]
        contrib_module_path = 'src/foo/contrib/'
        with mock_file_change_validator(
            path_content, self.pr_num, contrib_module_path
        ) as validator:
            result = validator.validate()
            self.assertTrue(result)


class FeatureApiValidatorTest(_CommonSetup, unittest.TestCase):

    def test_validation_failure_no_features_found(self):
        path_content = [
            ('readme.txt', None),
            ('src/foo/__init__.py', None),
            ('src/foo/contrib/__init__.py', None),
            ('src/foo/contrib/user_foo/__init__.py', None),
            ('src/foo/contrib/user_foo/feature_bar.py', None),
        ]
        contrib_module_path = 'src/foo/contrib/'
        with mock_feature_api_validator(
            path_content, self.pr_num, contrib_module_path, self.X, self.y
        ) as validator:
            result = validator.validate()
            self.assertFalse(result)

    def test_validation_failure_invalid_feature(self):
        path_content = [
            ('foo.jpg', None),
            ('src/foo/__init__.py', None),
            ('src/foo/contrib/__init__.py', None),
            ('src/foo/contrib/user_foo/__init__.py', None),
            ('src/foo/contrib/user_foo/feature_bar.py',
             self.invalid_feature_str),
        ]
        contrib_module_path = 'src/foo/contrib/'
        with mock_feature_api_validator(
            path_content, self.pr_num, contrib_module_path, self.X, self.y
        ) as validator:
            changes = validator.change_collector.collect_changes()
            self.assertEqual(len(changes.file_diffs), 1)
            self.assertEqual(len(changes.candidate_feature_diffs), 1)
            self.assertEqual(len(changes.valid_init_diffs), 0)
            self.assertEqual(len(changes.inadmissible_diffs), 0)

            # TODO
            # self.assertEqual(len(new_features), 1)
            # self.assertTrue(imported_okay)

            result = validator.validate()
            self.assertFalse(result)

    def test_validation_failure_import_error(self):
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
            path_content, self.pr_num, contrib_module_path, self.X, self.y
        ) as validator:
            changes = validator.change_collector.collect_changes()
            self.assertEqual(len(changes.file_diffs), 1)
            self.assertEqual(len(changes.candidate_feature_diffs), 1)
            self.assertEqual(len(changes.valid_init_diffs), 0)
            self.assertEqual(len(changes.inadmissible_diffs), 0)

            # TODO
            # self.assertEqual(len(new_feature_info), 0)
            # self.assertFalse(imported_okay)

            result = validator.validate()
            self.assertFalse(result)
