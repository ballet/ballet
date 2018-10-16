from copy import deepcopy

import numpy as np
from funcy import all_fn, isa, iterable

from ballet.feature import Feature
from ballet.validation.base import BaseCheck


class FeatureApiCheck(BaseCheck):
    """Base class for implementing new Feature API checks

    Args:
        X (array-like): X
        y (array-like): y
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y


class IsFeatureCheck(FeatureApiCheck):

    def check(self, feature):
        assert isinstance(feature, Feature)


class HasCorrectInputTypeCheck(FeatureApiCheck):

    def check(self, feature):
        """Check that `input` is a string or iterable of string"""
        input = feature.input
        is_str = isa(str)
        is_nested_str = all_fn(
            iterable, lambda x: all(map(is_str, x)))
        assert is_str(input) or is_nested_str(input)


class HasTransformerInterfaceCheck(FeatureApiCheck):

    def check(self, feature):
        assert hasattr(feature.transformer, 'fit')
        assert hasattr(feature.transformer, 'transform')
        assert hasattr(feature.transformer, 'fit_transform')


class CanMakeMapperCheck(FeatureApiCheck):

    def check(self, feature):
        feature.as_dataframe_mapper()


class CanFitCheck(FeatureApiCheck):

    def check(self, feature):
        mapper = feature.as_dataframe_mapper()
        mapper.fit(self.X, y=self.y)


class CanTransformCheck(FeatureApiCheck):

    def check(self, feature):
        mapper = feature.as_dataframe_mapper()
        mapper.fit(self.X, y=self.y)
        mapper.transform(self.X)


class CanFitTransformCheck(FeatureApiCheck):

    def check(self, feature):
        mapper = feature.as_dataframe_mapper()
        mapper.fit_transform(self.X, y=self.y)


class HasCorrectOutputDimensionsCheck(FeatureApiCheck):

    def check(self, feature):
        mapper = feature.as_dataframe_mapper()
        X = mapper.fit_transform(self.X, y=self.y)
        assert self.X.shape[0] == X.shape[0]


class CanDeepcopyCheck(FeatureApiCheck):

    def check(self, feature):
        deepcopy(feature)


class NoMissingValuesCheck(FeatureApiCheck):

    def check(self, feature):
        mapper = feature.as_dataframe_mapper()
        X = mapper.fit_transform(self.X, y=self.y)
        assert not np.any(np.isnan(X))


class NoInfiniteValuesCheck(FeatureApiCheck):

    def check(self, feature):
        mapper = feature.as_dataframe_mapper()
        X = mapper.fit_transform(self.X, y=self.y)
        assert not np.any(np.isinf(X))
