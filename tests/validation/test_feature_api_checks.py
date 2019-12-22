import unittest

import numpy as np

from ballet.compat import SimpleImputer
from ballet.eng.base import BaseTransformer
from ballet.eng.misc import IdentityTransformer
from ballet.feature import Feature
from ballet.util import has_nans
from ballet.validation.common import check_from_class
from ballet.validation.feature_api.checks import (
    CanDeepcopyCheck, CanTransformCheck, FeatureApiCheck,
    HasCorrectInputTypeCheck, HasCorrectOutputDimensionsCheck,
    NoMissingValuesCheck)

from .util import SampleDataMixin
from ..util import FragileTransformer


class FeatureApiCheckTest(SampleDataMixin, unittest.TestCase):

    def test_good_feature(self):
        feature = Feature(
            input='size',
            transformer=SimpleImputer(),
        )

        valid, failures = check_from_class(
            FeatureApiCheck, feature, self.X, self.y)
        self.assertTrue(valid)
        self.assertEqual(len(failures), 0)

    def test_bad_feature_input(self):
        # bad input
        feature = Feature(
            input=3,
            transformer=SimpleImputer(),
        )
        valid, failures = check_from_class(
            FeatureApiCheck, feature, self.X, self.y)
        self.assertFalse(valid)
        self.assertIn(HasCorrectInputTypeCheck.__name__, failures)

    def test_bad_feature_transform_errors(self):
        # transformer throws errors
        feature = Feature(
            input='size',
            transformer=FragileTransformer(
                (lambda x: True, ), (RuntimeError, ))
        )
        valid, failures = check_from_class(
            FeatureApiCheck, feature, self.X, self.y)
        self.assertFalse(valid)
        self.assertIn(CanTransformCheck.__name__, failures)

    def test_bad_feature_wrong_transform_length(self):
        class _WrongLengthTransformer(BaseTransformer):
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
        valid, failures = check_from_class(
            FeatureApiCheck, feature, self.X, self.y)
        self.assertFalse(valid)
        self.assertIn(HasCorrectOutputDimensionsCheck.__name__, failures)

    def test_bad_feature_deepcopy_fails(self):
        class _CopyFailsTransformer(IdentityTransformer):
            def __deepcopy__(self, memo):
                raise RuntimeError
        feature = Feature(
            input='size',
            transformer=_CopyFailsTransformer(),
        )
        valid, failures = check_from_class(
            FeatureApiCheck, feature, self.X, self.y)
        self.assertFalse(valid)
        self.assertIn(CanDeepcopyCheck.__name__, failures)

    def test_producing_missing_values_fails(self):
        assert has_nans(self.X)
        feature = Feature(
            input='size',
            transformer=IdentityTransformer()
        )
        valid, failures = check_from_class(
            FeatureApiCheck, feature, self.X, self.y)
        self.assertFalse(valid)
        self.assertIn(NoMissingValuesCheck.__name__, failures)
