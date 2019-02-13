from copy import deepcopy

import numpy as np
from funcy import all_fn, isa, iterable

from ballet.feature import Feature
from ballet.validation.base import BaseCheck


class FeatureApiCheck(BaseCheck):
    """Base class for implementing new Feature API checks

    Args:
        X_df (array-like): X dataframe
        y_df (array-like): y dataframe
    """

    def __init__(self, X_df, y_df):
        self.X = X_df
        self.y = y_df


class IsFeatureCheck(FeatureApiCheck):

    def check(self, feature):
        """Check that the object is an instance of ballet.Feature"""
        assert isinstance(feature, Feature)


class HasCorrectInputTypeCheck(FeatureApiCheck):

    def check(self, feature):
        """Check that the feature's `input` is a str or Iterable[str]"""
        input = feature.input
        is_str = isa(str)
        is_nested_str = all_fn(
            iterable, lambda x: all(map(is_str, x)))
        assert is_str(input) or is_nested_str(input)


class HasTransformerInterfaceCheck(FeatureApiCheck):

    def check(self, feature):
        """Check that the feature has a fit/transform/fit_tranform interface"""
        assert hasattr(feature.transformer, 'fit')
        assert hasattr(feature.transformer, 'transform')
        assert hasattr(feature.transformer, 'fit_transform')


class CanMakeMapperCheck(FeatureApiCheck):

    def check(self, feature):
        """Check that the feature can be converted to a DataFrameMapper"""
        feature.as_dataframe_mapper()


class CanFitCheck(FeatureApiCheck):

    def check(self, feature):
        """Check that fit can be called on reference data"""
        mapper = feature.as_dataframe_mapper()
        mapper.fit(self.X, y=self.y)


class CanTransformCheck(FeatureApiCheck):

    def check(self, feature):
        """Check that transform can be called on reference data"""
        mapper = feature.as_dataframe_mapper()
        mapper.fit(self.X, y=self.y)
        mapper.transform(self.X)


class CanFitTransformCheck(FeatureApiCheck):

    def check(self, feature):
        """Check that fit_transform can be called on reference data"""
        mapper = feature.as_dataframe_mapper()
        mapper.fit_transform(self.X, y=self.y)


class HasCorrectOutputDimensionsCheck(FeatureApiCheck):

    def check(self, feature):
        """Check that the dimensions of the transformed data are correct

        For input X, an n x p array, a n x q array should be produced,
        where q is the number of features produced by the logical feature.
        """
        mapper = feature.as_dataframe_mapper()
        X = mapper.fit_transform(self.X, y=self.y)
        assert self.X.shape[0] == X.shape[0]


class CanDeepcopyCheck(FeatureApiCheck):

    def check(self, feature):
        """Check that the feature can be deepcopied

        This is needed for execution of the overall transformation pipeline
        """
        deepcopy(feature)


class NoMissingValuesCheck(FeatureApiCheck):

    def check(self, feature):
        """Check that the output of the transformer has no missing values"""
        mapper = feature.as_dataframe_mapper()
        X = mapper.fit_transform(self.X, y=self.y)
        assert not np.any(np.isnan(X))


class NoInfiniteValuesCheck(FeatureApiCheck):

    def check(self, feature):
        """Check that the output of the transformer has no non-finite values"""
        mapper = feature.as_dataframe_mapper()
        X = mapper.fit_transform(self.X, y=self.y)
        assert not np.any(np.isinf(X))
