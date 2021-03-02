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
    NoMissingValuesCheck,)

from ..util import FragileTransformer


def test_good_feature(sample_data):
    feature = Feature(
        input='size',
        transformer=SimpleImputer(),
    )

    valid, failures, advice = check_from_class(
        FeatureApiCheck, feature, sample_data.X, sample_data.y)
    assert valid
    assert len(failures) == 0


def test_bad_feature_input(sample_data):
    # bad input
    feature = Feature(
        input=3,
        transformer=SimpleImputer(),
    )
    valid, failures, advice = check_from_class(
        FeatureApiCheck, feature, sample_data.X, sample_data.y)
    assert not valid
    assert HasCorrectInputTypeCheck.__name__ in failures


def test_bad_feature_transform_errors(sample_data):
    # transformer throws errors
    feature = Feature(
        input='size',
        transformer=FragileTransformer(
            (lambda x: True, ), (RuntimeError, ))
    )
    valid, failures, advice = check_from_class(
        FeatureApiCheck, feature, sample_data.X, sample_data.y)
    assert not valid
    assert CanTransformCheck.__name__ in failures


def test_bad_feature_wrong_transform_length(sample_data):
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
    valid, failures, advice = check_from_class(
        FeatureApiCheck, feature, sample_data.X, sample_data.y)
    assert not valid
    assert HasCorrectOutputDimensionsCheck.__name__ in failures


def test_bad_feature_deepcopy_fails(sample_data):
    class _CopyFailsTransformer(IdentityTransformer):
        def __deepcopy__(self, memo):
            raise RuntimeError
    feature = Feature(
        input='size',
        transformer=_CopyFailsTransformer(),
    )
    valid, failures, advice = check_from_class(
        FeatureApiCheck, feature, sample_data.X, sample_data.y)
    assert not valid
    assert CanDeepcopyCheck.__name__ in failures


def test_producing_missing_values_fails(sample_data):
    assert has_nans(sample_data.X)
    feature = Feature(
        input='size',
        transformer=IdentityTransformer()
    )
    valid, failures, advice = check_from_class(
        FeatureApiCheck, feature, sample_data.X, sample_data.y)
    assert not valid
    assert NoMissingValuesCheck.__name__ in failures
