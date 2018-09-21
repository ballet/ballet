import unittest
from textwrap import dedent
from unittest.mock import patch

import funcy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ballet.compat import SimpleImputer
from ballet.exc import UnexpectedValidationStateError
from ballet.feature import Feature
from ballet.util import IdentityTransformer, NoFitMixin
from ballet.util.gitutil import get_diff_str_from_commits
from ballet.util.travisutil import TravisPullRequestBuildDiffer
from ballet.validation import FeatureValidator, PullRequestFeatureValidator

from .util import (
    FragileTransformer, make_mock_commit, make_mock_commits, mock_repo)


class TestDataMixin:
    def setUp(self):
        self.df = pd.DataFrame(
            data={
                'country': ['USA', 'USA', 'Canada', 'Japan'],
                'year': [2001, 2002, 2001, 2002],
                'size': [np.nan, -11, 12, 0.0],
                'strength': [18, 110, np.nan, 101],
                'happy': [False, True, False, False]
            }
        ).set_index(['country', 'year'])
        self.X = self.df[['size', 'strength']]
        self.y = self.df[['happy']]
        super().setUp()


class TestFeatureValidator(TestDataMixin, unittest.TestCase):

    def test_good_feature(self):
        feature = Feature(
            input='size',
            transformer=SimpleImputer(),
        )

        validator = FeatureValidator(self.X, self.y)
        result, failures = validator.validate(feature)
        self.assertTrue(result)
        self.assertEqual(len(failures), 0)

    def test_bad_feature_input(self):
        # bad input
        feature = Feature(
            input=3,
            transformer=SimpleImputer(),
        )
        validator = FeatureValidator(self.X, self.y)
        result, failures = validator.validate(feature)
        self.assertFalse(result)
        self.assertIn('has_correct_input_type', failures)

    def test_bad_feature_transform_errors(self):
        # transformer throws errors
        feature = Feature(
            input='size',
            transformer=FragileTransformer(
                (lambda x: True, ), (RuntimeError, ))
        )
        validator = FeatureValidator(self.X, self.y)
        result, failures = validator.validate(feature)
        self.assertFalse(result)
        self.assertIn('can_transform', failures)

    def test_bad_feature_wrong_transform_length(self):
        class _WrongLengthTransformer(
                BaseEstimator, NoFitMixin, TransformerMixin):
            def transform(self, X, **transform_kwargs):
                new_shape = list(X.shape)
                new_shape[0] += 1
                output = np.arange(np.prod(new_shape)).reshape(new_shape)
                return output

        # doesn't return correct length
        feature = Feature(
            input='size',
            transformer=_WrongLengthTransformer(),
        )
        validator = FeatureValidator(self.X, self.y)
        result, failures = validator.validate(feature)
        self.assertFalse(result)
        self.assertIn('has_correct_output_dimensions', failures)

    def test_bad_feature_deepcopy_fails(self):
        class _CopyFailsTransformer(IdentityTransformer):
            def __deepcopy__(self):
                raise RuntimeError
        feature = Feature(
            input='size',
            transformer=_CopyFailsTransformer(),
        )
        validator = FeatureValidator(self.X, self.y)
        result, failures = validator.validate(feature)
        self.assertFalse(result)
        self.assertIn('can_deepcopy', failures)


class TestPullRequestFeatureValidator(TestDataMixin, unittest.TestCase):

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
        )
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
        )

    @funcy.contextmanager
    def null_prfv(self):
        with mock_repo() as repo:
            commit_range = 'HEAD^..HEAD'
            contrib_module_path = None
            X = None
            y = None

            travis_env_vars = {
                'TRAVIS_BUILD_DIR': repo.working_tree_dir,
                'TRAVIS_PULL_REQUEST': str(self.pr_num),
                'TRAVIS_COMMIT_RANGE': commit_range,
            }
            with patch.dict('os.environ', travis_env_vars):
                yield PullRequestFeatureValidator(
                    repo, self.pr_num, contrib_module_path, X, y)

    def test_prfv_init(self):
        with self.null_prfv() as validator:
            self.assertIsInstance(
                validator.differ, TravisPullRequestBuildDiffer)

    def test_prfv_required_method_ordering(self):
        with self.null_prfv() as validator:
            with self.assertRaises(UnexpectedValidationStateError):
                validator._categorize_file_diffs()
            with self.assertRaises(UnexpectedValidationStateError):
                validator._validate_files()
            with self.assertRaises(UnexpectedValidationStateError):
                validator._collect_features()
            with self.assertRaises(UnexpectedValidationStateError):
                validator._validate_features()
            with self.assertRaises(UnexpectedValidationStateError):
                validator._determine_validation_result()

    def test_prfv_collect_file_diffs(self):
        n = 10
        with mock_repo() as repo:
            commits = make_mock_commits(repo, n=n)
            contrib_module_path = None
            X = None
            y = None
            commit_range = get_diff_str_from_commits(
                commits[0], commits[-1])

            travis_env_vars = {
                'TRAVIS_BUILD_DIR': repo.working_tree_dir,
                'TRAVIS_PULL_REQUEST': str(self.pr_num),
                'TRAVIS_COMMIT_RANGE': commit_range,
            }
            with patch.dict('os.environ', travis_env_vars):
                validator = PullRequestFeatureValidator(
                    repo, self.pr_num, contrib_module_path, X, y)
                validator._collect_file_diffs()

                # checks on file_diffs
                self.assertEqual(len(validator.file_diffs), n - 1)
                for diff in validator.file_diffs:
                    self.assertEqual(diff.change_type, 'A')
                    self.assertTrue(diff.b_path.startswith('file'))
                    self.assertTrue(diff.b_path.endswith('.py'))

    @funcy.contextmanager
    def mock_project(self, path_content):
        with mock_repo() as repo:
            for path, content in path_content:
                make_mock_commit(repo, path=path, content=content)
            yield repo

    @funcy.contextmanager
    def _test_prfv_end_to_end(self, path_content, contrib_module_path):
        with self.mock_project(path_content) as repo:
            travis_build_dir = repo.working_tree_dir
            travis_pull_request = str(self.pr_num)
            travis_commit_range = get_diff_str_from_commits(
                repo.head.commit.parents[0], repo.head.commit)
            X, y = self.X, self.y

            travis_env_vars = {
                'TRAVIS_BUILD_DIR': travis_build_dir,
                'TRAVIS_PULL_REQUEST': travis_pull_request,
                'TRAVIS_COMMIT_RANGE': travis_commit_range,
            }
            with patch.dict('os.environ', travis_env_vars):
                yield PullRequestFeatureValidator(
                    repo, self.pr_num, contrib_module_path, X, y)

    def test_prfv_end_to_end_failure_no_features_found(self):
        path_content = [
            ('readme.txt', None),
            ('src/__init__.py', None),
            ('src/contrib/__init__.py', None),
            ('src/contrib/foo.py', None),
            ('src/contrib/baz.py', None),
        ]
        contrib_module_path = 'src/contrib/'
        with self._test_prfv_end_to_end(path_content, contrib_module_path) \
                as validator:
            result = validator.validate()
            self.assertFalse(result)

    def test_prfv_end_to_end_failure_inadmissible_file_diffs(self):
        path_content = [
            ('readme.txt', None),
            ('src/__init__.py', None),
            ('src/contrib/__init__.py', None),
            ('src/contrib/foo.py', None),
            ('invalid.py', None),
        ]
        contrib_module_path = 'src/contrib/'
        with self._test_prfv_end_to_end(path_content, contrib_module_path) \
                as validator:
            result = validator.validate()
            self.assertFalse(result)
            self.assertEqual(
                len(validator.file_diffs), 1)
            self.assertEqual(
                len(validator.file_diffs_admissible), 0)
            self.assertEqual(
                len(validator.file_diffs_inadmissible), 1)
            self.assertEqual(
                validator.file_diffs_inadmissible[0].b_path, 'invalid.py')
            self.assertFalse(
                validator.file_diffs_validation_result)

    def test_prfv_end_to_end_failure_bad_package_structure(self):
        path_content = [
            ('foo.jpg', None),
            ('src/contrib/bar/baz.py', self.valid_feature_str),
        ]
        contrib_module_path = 'src/contrib/'
        with self._test_prfv_end_to_end(path_content, contrib_module_path) \
                as validator:
            result = validator.validate()
            self.assertFalse(result)
            self.assertEqual(
                len(validator.file_diffs), 1)
            self.assertEqual(
                len(validator.file_diffs_admissible), 1)
            self.assertEqual(
                len(validator.file_diffs_inadmissible), 0)
            self.assertTrue(
                validator.file_diffs_validation_result)
            self.assertEqual(
                len(validator.features), 0)
            self.assertFalse(
                validator.features_validation_result)

    def test_prfv_end_to_end_failure_invalid_feature(self):
        path_content = [
            ('foo.jpg', None),
            ('src/__init__.py', None),
            ('src/contrib/__init__.py', None),
            ('src/contrib/foo.py', self.invalid_feature_str),
        ]
        contrib_module_path = 'src/contrib/'
        with self._test_prfv_end_to_end(path_content, contrib_module_path) \
                as validator:
            result = validator.validate()
            self.assertFalse(result)
            self.assertEqual(
                len(validator.file_diffs), 1)
            self.assertEqual(
                len(validator.file_diffs_admissible), 1)
            self.assertEqual(
                len(validator.file_diffs_inadmissible), 0)
            self.assertTrue(
                validator.file_diffs_validation_result)
            self.assertEqual(
                len(validator.features), 1)
            self.assertFalse(
                validator.features_validation_result)

    def test_prfv_end_to_end_success(self):
        path_content = [
            ('bob.xml', '<><> :: :)'),
            ('src/__init__.py', None),
            ('src/contrib/__init__.py', None),
            ('src/contrib/bean.py', self.valid_feature_str),
        ]
        contrib_module_path = 'src/contrib/'
        with self._test_prfv_end_to_end(path_content, contrib_module_path) \
                as validator:
            result = validator.validate()
            self.assertTrue(result)
