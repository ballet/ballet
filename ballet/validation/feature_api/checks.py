import io
from copy import deepcopy

import dill as pickle
import numpy as np
from funcy import all, all_fn, isa, iterable
from sklearn.model_selection import train_test_split

from ballet.feature import Feature
from ballet.util import RANDOM_STATE
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

    def give_advice(self, feature):
        return f'The object needs to be an instance of ballet.Feature, whereas it is actually of type {type(feature).__name__}'  # noqa


class HasCorrectInputTypeCheck(FeatureApiCheck):

    def check(self, feature):
        """Check that the feature's `input` is a str or Iterable[str]"""
        input = feature.input
        is_str = isa(str)
        is_nested_str = all_fn(
            iterable, lambda x: all(is_str, x))
        assert is_str(input) or is_nested_str(input)

    def give_advice(self, feature):
        return f'The feature\'s input needs to be a string or list of strings, whereas it is actually of type {type(feature.input).__name__}'  # noqa


class HasTransformerInterfaceCheck(FeatureApiCheck):

    def check(self, feature):
        """Check that the feature has a fit/transform/fit_tranform interface"""
        assert hasattr(feature.transformer, 'fit')
        assert hasattr(feature.transformer, 'transform')
        assert hasattr(feature.transformer, 'fit_transform')

    def give_advice(self, feature):
        missing = ', '.join(
            attr
            for attr in ('fit', 'transform', 'fit_transform')
            if not hasattr(feature.transformer, attr)
        )
        return f'The feature\'s transformer must have the transformer interface, but these methods are missing and must be implemented: {missing}'  # noqa


class CanMakeMapperCheck(FeatureApiCheck):

    def check(self, feature):
        """Check that the feature can be converted to a FEP"""
        feature.as_feature_engineering_pipeline()

    def give_advice(self, feature):
        return 'The following method call fails and needs to be fixed: feature.as_feature_engineering_pipeline()'  # noqa


class CanFitCheck(FeatureApiCheck):

    def check(self, feature):
        """Check that fit can be called on reference data"""
        mapper = feature.as_feature_engineering_pipeline()
        mapper.fit(self.X, y=self.y)

    def give_advice(self, feature):
        return 'The feature fails when calling fit on sample data'


class CanFitOneRowCheck(FeatureApiCheck):

    def check(self, feature):
        """Check that fit can be called on one row of reference data"""
        mapper = feature.as_feature_engineering_pipeline()
        x, y = _get_one_row(self.X, self.y)
        mapper.fit(x, y=y)

    def give_advice(self, feature):
        return 'The feature fails when calling fit on a single row of sample data (i.e. 1xn array)'  # noqa


class CanTransformCheck(FeatureApiCheck):

    def check(self, feature):
        """Check that transform can be called on reference data"""
        mapper = feature.as_feature_engineering_pipeline()
        mapper.fit(self.X, y=self.y)
        mapper.transform(self.X)

    def give_advice(self, feature):
        return 'The feature fails when calling transform on sample data'


class CanTransformNewRowsCheck(FeatureApiCheck):

    def check(self, feature):
        """Check that transform can be called on new, unseen rows"""
        mapper = feature.as_feature_engineering_pipeline()
        X1, X2, y1, _ = train_test_split(
            self.X, self.y, test_size=0.1, random_state=RANDOM_STATE,
            shuffle=True)
        mapper.fit(X1, y=y1)
        mapper.transform(X2)

    def give_advice(self, feature):
        return 'The feature fails when calling transform on different data than it was trained on; make sure the transform method works on any number of new rows'  # noqa


class CanTransformOneRowCheck(FeatureApiCheck):

    def check(self, feature):
        """Check that transform can be called on one row of reference data"""
        mapper = feature.as_feature_engineering_pipeline()
        mapper.fit(self.X, y=self.y)
        x, = _get_one_row(self.X)
        mapper.transform(x)

    def give_advice(self, feature):
        return 'The feature fails when calling transform on a single row of sample data (i.e. 1xn array)'  # noqa


class CanFitTransformCheck(FeatureApiCheck):

    def check(self, feature):
        """Check that fit_transform can be called on reference data"""
        mapper = feature.as_feature_engineering_pipeline()
        mapper.fit_transform(self.X, y=self.y)

    def give_advice(self, feature):
        return 'The feature fails when calling fit_transform on sample data'


class HasCorrectOutputDimensionsCheck(FeatureApiCheck):

    def check(self, feature):
        """Check that the dimensions of the transformed data are correct

        For input X, an n x p array, a n x q array should be produced,
        where q is the number of feature values produced by the feature.
        """
        mapper = feature.as_feature_engineering_pipeline()
        X = mapper.fit_transform(self.X, y=self.y)
        assert self.X.shape[0] == X.shape[0]

    def give_advice(self, feature):
        mapper = feature.as_feature_engineering_pipeline()
        X = mapper.fit_transform(self.X, y=self.y)
        n = self.X.shape[0]
        m = X.shape[0]

        return f'The feature does not produce the correct output dimensions, for example when it is fit and transformed on {n} rows of data, it produces {m} rows of feature values.'  # noqa


class CanDeepcopyCheck(FeatureApiCheck):

    def check(self, feature):
        """Check that the feature can be deepcopied

        This is needed for execution of the overall transformation pipeline
        """
        deepcopy(feature)

    def give_advice(self, feature):
        return 'Calling copy.deepcopy(feature) fails, make sure every component of the feature can be deepcopied'  # noqa


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

    def give_advice(self, feature):
        return 'Calling pickle.dump(feature, buf) fails, make sure the feature can be pickled'  # noqa


class NoMissingValuesCheck(FeatureApiCheck):

    def check(self, feature):
        """Check that the output of the transformer has no missing values"""
        mapper = feature.as_feature_engineering_pipeline()
        X = mapper.fit_transform(self.X, y=self.y)
        assert not np.any(np.isnan(X))

    def give_advice(self, feature):
        return 'When transforming sample data, the feature produces NaN values. If you reasonably expect these missing values, make sure you clean missing values as an additional step in your transformer list. For example: NullFiller(replacement=replacement)'  # noqa


class NoInfiniteValuesCheck(FeatureApiCheck):

    def check(self, feature):
        """Check that the output of the transformer has no non-finite values"""
        mapper = feature.as_feature_engineering_pipeline()
        X = mapper.fit_transform(self.X, y=self.y)
        assert not np.any(np.isinf(X))

    def give_advice(self, feature):
        return 'When transforming sample data, the feature produces infinite values. You can detect these with np.isinf. If you reasonably expect these infinite values, make sure you clean infinite values as an additional step in your transformer list. For example: NullFiller(np.isinf, replacement) '  # noqa
