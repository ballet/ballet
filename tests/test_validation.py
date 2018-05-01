import unittest
from unittest.mock import patch

import funcy
import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn.base import BaseEstimator, TransformerMixin

from fhub_core.exc import UnexpectedValidationStateError
from fhub_core.feature import Feature
from fhub_core.util import IdentityTransformer, NoFitMixin
from fhub_core.util.gitutil import get_diff_str_from_commits
from fhub_core.util.travisutil import TravisPullRequestBuildDiffer
from fhub_core.validation import FeatureValidator, PullRequestFeatureValidator

from .util import (
    FragileTransformer, make_mock_commit, make_mock_commits, mock_repo)


class TestFeatureValidator(unittest.TestCase):
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

    def test_good_feature(self):
        feature = Feature(
            input='size',
            transformer=sklearn.preprocessing.Imputer(),
        )

        validator = FeatureValidator(self.X, self.y)
        result, failures = validator.validate(feature)
        self.assertTrue(result)
        self.assertEqual(len(failures), 0)

    def test_bad_feature_input(self):
        # bad input
        feature = Feature(
            input=3,
            transformer=sklearn.preprocessing.Imputer(),
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


class TestPullRequestFeatureValidator(unittest.TestCase):

    def setUp(self):
        self.pr_num = 73

    @unittest.expectedFailure
    def test_todo(self):
        raise NotImplementedError

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
                    self.pr_num, contrib_module_path, X, y)

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
                    self.pr_num, contrib_module_path, X, y)
                validator._collect_file_diffs()

                # checks on file_diffs
                self.assertEqual(len(validator.file_diffs), n - 1)
                for diff in validator.file_diffs:
                    self.assertEqual(diff.change_type, 'A')
                    self.assertTrue(diff.b_path.startswith('file'))
                    self.assertTrue(diff.b_path.endswith('.py'))
