import sys
import traceback

import numpy as np
import pandas as pd
from funcy import identity, is_seqcont, wraps
from sklearn.pipeline import _name_estimators
from sklearn_pandas import DataFrameMapper
from sklearn_pandas.pipeline import TransformerPipeline, make_transformer_pipeline

from ballet.util import asarray2d, indent
from ballet.util.log import logger

__all__ = ['make_mapper', 'Feature']


def make_mapper(features):
    """Make a DataFrameMapper from a list of features

    Args:
        features (List[Feature]): list of features

    Returns:
        DataFrameMapper: mapper made from features
    """
    return DataFrameMapper(
        [t.as_input_transformer_tuple() for t in features],
        input_df=True)


default_caught = (ValueError, TypeError)
conversion_approaches = [
    ('identity', identity, default_caught),
    ('series', pd.Series, default_caught),
    ('dataframe', pd.DataFrame, default_caught),
    ('array', np.asarray, default_caught),
    ('asarray2d', asarray2d, ()),
]


def log_failure_detail(failure_detail):
    print('*** logging is happening', flush=True)
    logger.info("Conversion failed, and we're not sure why...")
    logger.info('Here are the approaches we tried, and how they failed.')
    for i, item in enumerate(failure_detail):
        msg = ('{i}. Conversion using approach {name!r} resulted in '
               '{exc_name} ({exc}), ')
        if item['outcome'] == 'caught':
            msg += 'which was caught, and the next approach was tried.'
        elif item['outcome'] == 'failure':
            msg += 'which was an unexpected error resulting in failure.'
        else:
            raise RuntimeError
        exc_name = type(item['exc']).__name__
        logger.info(msg.format(i=i+1, exc_name=exc_name, **item))


def make_robust_to_tabular_types(func):

    @wraps(func)
    def wrapped(X, y=None, **kwargs):
        failure_detail = []
        for name, convert, caught in conversion_approaches:
            try:
                logger.debug(
                    "Converting using approach '{name}'"
                    .format(name=name))
                if y is not None:
                    return func(convert(X), y=convert(y), **kwargs)
                else:
                    return func(convert(X), **kwargs)
            except caught as e:
                tb = traceback.format_exc()
                failure_detail.append(
                    {'name': name, 'exc': e, 'tb': tb, 'outcome': 'caught'})
                pretty_tb = indent(tb, n=8)
                logger.debug(
                    "Conversion approach didn't work, "
                    "caught exception '{name}'\n\n{tb}"
                    .format(name=type(e).__name__, tb=pretty_tb))
                continue
            except Exception as e:
                tb = traceback.format_exc()
                failure_detail.append(
                    {'name': name, 'exc': e, 'tb': tb, 'outcome': 'failure'})
                log_failure_detail(failure_detail)
                raise

        # none of the conversion attempts succeeded
        log_failure_detail(failure_detail)

    return wrapped


def make_robust_transformer(transformer):
    transformer.fit = make_robust_to_tabular_types(transformer.fit)
    transformer.transform = make_robust_to_tabular_types(transformer.transform)
    return transformer


def make_robust_transformer_pipeline(*steps):
    '''Construct a RobustTransformerPipeline from the given estimators.'''
    transformer_pipeline = make_transformer_pipeline(*steps)
    for i in range(transformer_pipeline.steps):
        transformer_pipeline.steps[i][1] = make_robust_transformer(transformer_pipeline.steps[i][1])
    return transformer_pipeline


class Feature:
    def __init__(self, input, transformer, name=None, description=None,
                 output=None, source=None, options=None):
        self.input = input
        if is_seqcont(transformer):
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
