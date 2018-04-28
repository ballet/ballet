import copy
import logging
import traceback

import funcy
import numpy as np
import pandas as pd
from sklearn.pipeline import _name_estimators
from sklearn_pandas import DataFrameMapper
from sklearn_pandas.pipeline import TransformerPipeline

from fhub_core.util import asarray2d, indent

logger = logging.getLogger(__name__)


__all__ = ['Feature', 'FeatureValidator', 'make_robust_transformer',
           'RobustTransformerPipeline', 'make_robust_transformer_pipeline']


class RobustTransformerPipeline(TransformerPipeline):

    def transform(self, X, **transform_kwargs):
        _transform = make_robust_to_tabular_types(super().transform)
        return _transform(X, **transform_kwargs)


def make_robust_transformer_pipeline(*steps):
    '''Construct a RobustTransformerPipeline from the given estimators.'''
    return RobustTransformerPipeline(_name_estimators(steps))


def make_conversion_approaches():
    funcs = (funcy.identity, pd.Series, pd.DataFrame, np.asarray, asarray2d)
    catch = (ValueError, TypeError)
    for func in funcs[:-1]:
        yield func, catch
    yield funcs[-1], ()


def make_robust_to_tabular_types(func):
    @funcy.wraps(func)
    def wrapped(X, y=None, **kwargs):
        for convert, catch in make_conversion_approaches():
            try:
                logger.debug(
                    "Converting using approach '{}'".format(convert.__name__))
                if y is not None:
                    return func(convert(X), y=convert(y), **kwargs)
                else:
                    return func(convert(X), **kwargs)
            except catch as e:
                formatted_exc = indent(traceback.format_exc(), n=8)
                logger.debug(
                    "Application subsequently failed with exception '{}'\n\n{}"
                    .format(e.__class__.__name__, formatted_exc))
    return wrapped


def make_robust_transformer(transformer):
    transformer.fit = make_robust_to_tabular_types(transformer.fit)
    # todo optionally catch all errors of transformer and return []
    transformer.transform = make_robust_to_tabular_types(transformer.transform)
    return transformer


class Feature:
    def __init__(self, input, transformer, name=None, description=None,
            output=None, source=None, options=None):
        self.input = input
        if funcy.is_seqcont(transformer):
            transformer = make_robust_transformer_pipeline(*transformer)
        self.transformer = make_robust_transformer(transformer)
        self.name = name
        self.description = description
        self.output = output  # unused
        self.source = source
        self.options = options if options is not None else {}

    def __repr__(self):
        # TODO use self.__dict__ directly, which respects insertion order
        attr_list = ['input', 'transformer', 'name', 'description', 'output',
            'source', 'options']
        attrs_str = ', '.join(
            '{attr_name}={attr_val}'.format(
                attr_name=attr, attr_val = getattr(self, attr)
            ) for attr in attr_list
        )
        return self.__class__.__name__ + '(' + attrs_str + ')'

    def as_input_transformer_tuple(self):
        return (self.input, self.transformer)

    def as_dataframe_mapper(self):
        return DataFrameMapper([
            self.as_input_transformer_tuple()
        ], input_df=True)


def check(func):
    '''Evaluate func, returning T if no errors and F if AssertionError'''
    @funcy.wraps(func)
    def wrapped(*args, **kwargs):
        try:
            func(*args, **kwargs)
            return True
        except AssertionError:
            return False
    wrapped.is_check = True
    return wrapped


class FeatureValidator:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    @check
    def is_feature(self, feature):
        assert isinstance(feature, Feature)

    @check
    def has_correct_input_type(self, feature):
        '''Check that `input` is a string or iterable of string'''
        input = feature.input
        is_str = funcy.isa(str)
        is_nested_str = funcy.all_fn(
            funcy.iterable, lambda x: all(map(is_str, x)))
        assert is_str(input) or is_nested_str(input)

    @check
    def has_transformer_interface(self, feature):
        assert hasattr(feature.transformer, 'fit')
        assert hasattr(feature.transformer, 'transform')

    @check
    def can_make_mapper(self, feature):
        try:
            mapper = feature.as_dataframe_mapper()
        except Exception:
            raise AssertionError

    @check
    def can_fit(self, feature):
        try:
            mapper = feature.as_dataframe_mapper()
            mapper.fit(self.X, self.y)
        except Exception:
            raise AssertionError

    @check
    def can_transform(self, feature):
        try:
            mapper = feature.as_dataframe_mapper()
            mapper.fit(self.X, self.y)
            mapper.transform(self.X)
        except Exception:
            raise AssertionError

    @check
    def can_fit_transform(self, feature):
        try:
            mapper = feature.as_dataframe_mapper()
            mapper.fit_transform(self.X, self.y)
        except Exception:
            raise AssertionError

    @check
    def has_correct_output_dimensions(self, feature):
        try:
            mapper = feature.as_dataframe_mapper()
            X = mapper.fit_transform(self.X, self.y)
        except Exception:
            raise AssertionError

        assert self.X.shape[0] == X.shape[0]

    @check
    def can_deepcopy(self, feature):
        try:
            copy.deepcopy(feature)
        except Exception:
            raise AssertionError

    def get_all_checks(self):
        for method_name in self.__dir__():
            method = getattr(self, method_name)
            if hasattr(method, 'is_check') and method.is_check:
                name = method.__name__
                yield (method, name)

    def validate(self, feature):
        failures = []
        result = True
        for check, name in self.get_all_checks():
            success = check(feature)
            if not success:
                result = False
                failures.append(name)

        return result, failures
