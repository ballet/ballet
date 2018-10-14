import unittest
from textwrap import dedent
from unittest.mock import Mock, create_autospec, patch

from ballet.compat import pathlib
from ballet.project import Project
from ballet.util.ci import TravisPullRequestBuildDiffer
from ballet.util.git import get_diff_str_from_commits
from ballet.validation.project_structure import (
    relative_to_contrib,
    IsAdditionCheck, IsPythonSourceCheck, WithinContribCheck,
    SubpackageNameCheck, ModuleNameCheck, RelativeNameDepthCheck,
    ChangeCollector)

from ..util import make_mock_commits, mock_repo
from .util import (
    SampleDataMixin, make_mock_project, mock_feature_api_validator,
    mock_file_change_validator, null_change_collector)


class CommonSetup(SampleDataMixin):

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


class DiffCheckTest(unittest.TestCase):

    def setUp(self):
        self.contrib_module_path = 'foo/features/contrib'
        self.project = create_autospec(
            Project, contrib_module_path=self.contrib_module_path)

    def test_relative_to_contrib(self):
        diff = Mock(b_path='foo/features/contrib/abc.py')
        project = self.project

        expected = pathlib.Path('abc.py')
        actual = relative_to_contrib(diff, project)
        self.assertEqual(actual, expected)

    def test_is_addition_check(self):
        checker = IsAdditionCheck(self.project)

        mock_diff = Mock(change_type='A')
        self.assertTrue(checker.do_check(mock_diff))

        mock_diff = Mock(change_type='B')
        self.assertFalse(checker.do_check(mock_diff))

    def test_is_python_source_check(self):
        checker = IsPythonSourceCheck(self.project)

        mock_diff = Mock(b_path='foo/features/contrib/user_bob/feature_1.py')
        self.assertTrue(checker.do_check(mock_diff))

        mock_diff = Mock(b_path='foo/features/contrib/user_bob/feature_1.xyz')
        self.assertFalse(checker.do_check(mock_diff))

    def test_within_contrib_check(self):
        checker = WithinContribCheck(self.project)

        mock_diff = Mock(b_path='foo/features/contrib/user_bob/feature_1.py')
        self.assertTrue(checker.do_check(mock_diff))

        mock_diff = Mock(b_path='foo/hack.py')
        self.assertFalse(checker.do_check(mock_diff))

    def test_subpackage_name_check(self):
        checker = SubpackageNameCheck(self.project)

        mock_diff = Mock(b_path='foo/features/contrib/user_bob/feature_1.py')
        self.assertTrue(checker.do_check(mock_diff))

        mock_diff = Mock(b_path='foo/features/contrib/bob/feature_1.py')
        self.assertFalse(checker.do_check(mock_diff))

    def test_feature_module_name_check(self):
        checker = ModuleNameCheck(self.project)

        mock_diff = Mock(b_path='foo/features/contrib/user_bob/feature_1.py')
        self.assertTrue(checker.do_check(mock_diff))

        bad_paths = [
            'foo/features/contrib/user_bob/foo1.py',
            'foo/features/contrib/user_bob/1.py',
            'foo/features/contrib/user_bob/feature_x-1.py',
        ]
        for path in bad_paths:
            mock_diff = Mock(b_path=path)
            self.assertFalse(checker.do_check(mock_diff))

    def test_relative_name_depth_check(self):
        checker = RelativeNameDepthCheck(self.project)

        mock_diff = Mock(b_path='foo/features/contrib/user_bob/feature_1.py')
        self.assertTrue(checker.do_check(mock_diff))

        mock_diff = Mock(b_path='foo/features/contrib/user_bob/a/b/c/d/feature_1.py')
        self.assertFalse(checker.do_check(mock_diff))


class ChangeCollectorTest(CommonSetup, unittest.TestCase):

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
            commit_range = get_diff_str_from_commits(
                commits[0], commits[-1])

            travis_env_vars = {
                'TRAVIS_BUILD_DIR': repo.working_tree_dir,
                'TRAVIS_PULL_REQUEST': str(self.pr_num),
                'TRAVIS_COMMIT_RANGE': commit_range,
            }

            with patch.dict('os.environ', travis_env_vars, clear=True):
                project = make_mock_project(repo, self.pr_num,
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


class FileChangeValidatorTest(CommonSetup, unittest.TestCase):

    def test_validation_failure_inadmissible_file_diffs(self):
        path_content = [
            ('readme.txt', None),
            ('src/__init__.py', None),
            ('src/contrib/__init__.py', None),
            ('src/contrib/user_foo/feature_bar.py', None),
            ('src/contrib/user_foo/__init__.py', None),
            ('invalid.py', None),
        ]
        contrib_module_path = 'src/contrib/'
        with mock_file_change_validator(
            path_content, self.pr_num, contrib_module_path
        ) as validator:
            collected_changes = validator.change_collector.collect_changes()
            self.assertEqual(len(collected_changes.file_diffs), 1)
            self.assertEqual(len(collected_changes.candidate_feature_diffs), 0)
            self.assertEqual(len(collected_changes.valid_init_diffs), 0)
            self.assertEqual(len(collected_changes.inadmissible_diffs), 1)
            self.assertEqual(
                collected_changes.inadmissible_diffs[0].b_path, 'invalid.py')

            # TODO
            # self.assertTrue(imported_okay)

            result = validator.validate()
            self.assertFalse(result)

    def test_validation_success(self):
        path_content = [
            ('bob.xml', '<hello>'),
            ('src/__init__.py', None),
            ('src/contrib/__init__.py', None),
            ('src/contrib/user_foo/__init__.py', None),
            ('src/contrib/user_foo/feature_bar.py', None),
        ]
        contrib_module_path = 'src/contrib/'
        with mock_file_change_validator(
            path_content, self.pr_num, contrib_module_path
        ) as validator:
            result = validator.validate()
            self.assertTrue(result)


class FeatureApiValidatorTest(CommonSetup, unittest.TestCase):

    def test_validation_failure_no_features_found(self):
        path_content = [
            ('readme.txt', None),
            ('src/__init__.py', None),
            ('src/contrib/__init__.py', None),
            ('src/contrib/user_foo/__init__.py', None),
            ('src/contrib/user_foo/feature_bar.py', None),
        ]
        contrib_module_path = 'src/contrib/'
        with mock_feature_api_validator(
            path_content, self.pr_num, contrib_module_path, self.X, self.y
        ) as validator:
            result = validator.validate()
            self.assertFalse(result)

    def test_validation_failure_invalid_feature(self):
        path_content = [
            ('foo.jpg', None),
            ('src/__init__.py', None),
            ('src/contrib/__init__.py', None),
            ('src/contrib/user_foo/__init__.py', None),
            ('src/contrib/user_foo/feature_bar.py', self.invalid_feature_str),
        ]
        contrib_module_path = 'src/contrib/'
        with mock_feature_api_validator(
            path_content, self.pr_num, contrib_module_path, self.X, self.y
        ) as validator:
            collected_changes = validator.change_collector.collect_changes()
            self.assertEqual(len(collected_changes.file_diffs), 1)
            self.assertEqual(len(collected_changes.candidate_feature_diffs), 1)
            self.assertEqual(len(collected_changes.valid_init_diffs), 0)
            self.assertEqual(len(collected_changes.inadmissible_diffs), 0)

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
            ('src/__init__.py', None),
            ('src/contrib/__init__.py', None),
            ('src/contrib/user_foo/__init__.py', None),
            ('src/contrib/user_foo/feature_baz.py', import_error_str),
        ]
        contrib_module_path = 'src/contrib/'
        with mock_feature_api_validator(
            path_content, self.pr_num, contrib_module_path, self.X, self.y
        ) as validator:
            collected_changes = validator.change_collector.collect_changes()
            self.assertEqual(len(collected_changes.file_diffs), 1)
            self.assertEqual(len(collected_changes.candidate_feature_diffs), 1)
            self.assertEqual(len(collected_changes.valid_init_diffs), 0)
            self.assertEqual(len(collected_changes.inadmissible_diffs), 0)

            # TODO
            # self.assertEqual(len(new_feature_info), 0)
            # self.assertFalse(imported_okay)

            result = validator.validate()
            self.assertFalse(result)
