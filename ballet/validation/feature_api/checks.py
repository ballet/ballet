import io
from copy import deepcopy

import dill as pickle
import numpy as np
from funcy import all, all_fn, isa, iterable

from ballet.feature import Feature
from ballet.validation.base import BaseCheck


def _get_one_row(*args):
    return tuple(
        obj.iloc[0:1]
        for obj in args
    )


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
            iterable, lambda x: all(is_str, x))
        assert is_str(input) or is_nested_str(input)


class HasTransformerInterfaceCheck(FeatureApiCheck):

    def check(self, feature):
        """Check that the feature has a fit/transform/fit_tranform interface"""
        assert hasattr(feature.transformer, 'fit')
        assert hasattr(feature.transformer, 'transform')
        assert hasattr(feature.transformer, 'fit_transform')


class CanMakeMapperCheck(FeatureApiCheck):

    def check(self, feature):
        """Check that the feature can be converted to a FEP"""
        feature.as_feature_engineering_pipeline()


class CanFitCheck(FeatureApiCheck):

    def check(self, feature):
        """Check that fit can be called on reference data"""
        mapper = feature.as_feature_engineering_pipeline()
        mapper.fit(self.X, y=self.y)


class CanFitOneRowCheck(FeatureApiCheck):

    def check(self, feature):
        """Check that fit can be called on one row of reference data"""
        mapper = feature.as_feature_engineering_pipeline()
        x, y = _get_one_row(self.X, self.y)
        mapper.fit(x, y=y)


class CanTransformCheck(FeatureApiCheck):

    def check(self, feature):
        """Check that transform can be called on reference data"""
        mapper = feature.as_feature_engineering_pipeline()
        mapper.fit(self.X, y=self.y)
        mapper.transform(self.X)


class CanTransformOneRowCheck(FeatureApiCheck):

    def check(self, feature):
        """Check that transform can be called on one row of reference data"""
        mapper = feature.as_feature_engineering_pipeline()
        mapper.fit(self.X, y=self.y)
        x, = _get_one_row(self.X)
        mapper.transform(x)


class CanFitTransformCheck(FeatureApiCheck):

    def check(self, feature):
        """Check that fit_transform can be called on reference data"""
        mapper = feature.as_feature_engineering_pipeline()
        mapper.fit_transform(self.X, y=self.y)


class HasCorrectOutputDimensionsCheck(FeatureApiCheck):

    def check(self, feature):
        """Check that the dimensions of the transformed data are correct

        For input X, an n x p array, a n x q array should be produced,
        where q is the number of features produced by the logical feature.
        """
        mapper = feature.as_feature_engineering_pipeline()
        X = mapper.fit_transform(self.X, y=self.y)
        assert self.X.shape[0] == X.shape[0]


class CanDeepcopyCheck(FeatureApiCheck):

    def check(self, feature):
        """Check that the feature can be deepcopied

        This is needed for execution of the overall transformation pipeline
        """
        deepcopy(feature)


class CanPickleCheck(FeatureApiCheck):

    def check(self, feature):
        """Check that the feature can be pickled

        This is needed for saving the pipeline to disk
        """
        try:
            buf = io.BytesIO()
            pickle.dump(feature, buf, protocol=pickle.HIGHEST_PROTOCOL)
            buf.seek(0)
            new_feature = pickle.load(buf)
            assert new_feature is not None
            assert isinstance(new_feature, Feature)
        finally:
            buf.close()


class NoMissingValuesCheck(FeatureApiCheck):

    def check(self, feature):
        """Check that the output of the transformer has no missing values"""
        mapper = feature.as_feature_engineering_pipeline()
        X = mapper.fit_transform(self.X, y=self.y)
        assert not np.any(np.isnan(X))


class NoInfiniteValuesCheck(FeatureApiCheck):

    def check(self, feature):
        """Check that the output of the transformer has no non-finite values"""
        mapper = feature.as_feature_engineering_pipeline()
        X = mapper.fit_transform(self.X, y=self.y)
        assert not np.any(np.isinf(X))
