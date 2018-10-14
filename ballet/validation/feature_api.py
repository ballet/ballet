from abc import ABCMeta, abstractmethod
from copy import deepcopy

import numpy as np
from funcy import all_fn, constantly, ignore, isa, iterable, post_processing

from ballet.feature import Feature
from ballet.util import whether_failures


@whether_failures
def validate(feature, X, y):
    """Validate the feature"""
    for check, name in get_checks(X, y):
        success = check(feature)
        if not success:
            yield name


def get_checks(X, y):
    checks = FeatureApiCheck.__subclasses__()
    for Checker in checks:
        method = Checker(X, y).do_check
        name = Checker.__name__
        yield method, name


class FeatureApiCheck(metaclass=ABCMeta):
    """Base class for implementing new Feature API checks

    To add a new check, simply subclass this class and implement the ``check``
    method.

    Args:
        X: X
        y: y
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y

    @ignore(Exception, default=False)
    @post_processing(constantly(True))
    def do_check(self, feature):
        self.check(feature)

    @abstractmethod
    def check(self, feature):
        """Check something about the feature, raising an error on failure

        Note that the return value of this method is ignored; the check fails
        if any Exception is raised, and succeeds otherwise.

        Args:
            feature (Feature): the feature to check

        Raises:
            Exception: the provided feature failed the API check
        """
        pass


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
