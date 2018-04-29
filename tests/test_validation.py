import unittest

import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn.base import BaseEstimator, TransformerMixin

from fhub_core.feature import Feature
from fhub_core.tmp import IdentityTransformer, NoFitMixin
from fhub_core.validation import FeatureValidator

from .util import FragileTransformer


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
        pass

    @unittest.expectedFailure
    def test_todo(self):
        raise NotImplementedError
