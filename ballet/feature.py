import traceback

import funcy
import numpy as np
import pandas as pd
from sklearn.pipeline import _name_estimators
from sklearn_pandas import DataFrameMapper
from sklearn_pandas.pipeline import TransformerPipeline

from ballet.util import asarray2d, indent
from ballet.util.log import logger


__all__ = ['Feature', 'make_robust_transformer',
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
            '{attr_name}={attr_val!r}'.format(
                attr_name=attr, attr_val=getattr(self, attr)
            ) for attr in attr_list
        )
        return self.__class__.__name__ + '(' + attrs_str + ')'

    def as_input_transformer_tuple(self):
        return (self.input, self.transformer)

    def as_dataframe_mapper(self):
        return DataFrameMapper([
            self.as_input_transformer_tuple()
        ], input_df=True)
