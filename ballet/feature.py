from collections import Counter, namedtuple
import traceback

import numpy as np
import pandas as pd
from funcy import identity, is_seqcont, select_values
from sklearn.base import TransformerMixin
from sklearn_pandas import DataFrameMapper
from sklearn_pandas.pipeline import TransformerPipeline

from ballet.exc import UnsuccessfulInputConversionError
from ballet.util import asarray2d, indent, DeepcopyMixin
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


def _name_estimators(estimators):
    """Generate names for estimators.

    Adapted from sklearn.pipeline._name_estimators
    """

    def get_name(estimator):
        if isinstance(estimator, DelegatingRobustTransformer):
            return get_name(estimator._transformer)

        return type(estimator).__name__.lower()

    names = list(map(get_name, estimators))
    counter = dict(Counter(names))
    counter = select_values(lambda x: x>1 , counter)

    for i in reversed(range(len(estimators))):
        name = names[i]
        if name in counter:
            names[i] += "-%d" % counter[name]
            counter[name] -= 1

    return list(zip(names, estimators))


def make_transformer_pipeline(*steps):
    """Construct a TransformerPipeline from the given estimators.

    Source: sklearn_pandas.make_transformer_pipeline
    """
    return TransformerPipeline(_name_estimators(steps))


def make_robust_transformer_pipeline(*steps):
    '''Construct a RobustTransformerPipeline from the given estimators.'''
    steps = list(map(DelegatingRobustTransformer, steps))
    return make_transformer_pipeline(*steps)


ConversionApproach = namedtuple('ConversionApproach', 'name convert caught')


class DelegatingRobustTransformer(DeepcopyMixin, TransformerMixin):

    DEFAULT_CAUGHT = (ValueError, TypeError)

    CONVERSION_APPROACHES = [
        ConversionApproach('identity', identity, DEFAULT_CAUGHT),
        ConversionApproach('series', pd.Series, DEFAULT_CAUGHT),
        ConversionApproach('dataframe', pd.DataFrame, DEFAULT_CAUGHT),
        ConversionApproach('array', np.asarray, DEFAULT_CAUGHT),
        ConversionApproach('asarray2d', asarray2d, ()),
    ]

    def __init__(self, transformer):
        self._transformer = transformer
        self._stored_conversion_approach = None

    def __getattr__(self, attr):
        return getattr(self._transformer, attr)

    def __repr__(self):
        name = type(self).__name__
        return '{name}({transformer!r})'.format(
            name=name, transformer=self._transformer)

    def fit(self, X, y=None, **kwargs):
        # don't return the result of transformer.fit because it is the
        # underlying transformer, not this robust transformer
        self._call_robust(self._transformer.fit, X, y, kwargs)

        # instead, return this robust transformer
        return self

    def transform(self, X, y=None, **kwargs):
        return self._call_robust(self._transformer.transform, X, y, kwargs)

    @staticmethod
    def _call_with_convert(method, convert, X, y, kwargs):
        if y is not None:
            return method(convert(X), y=convert(y), **kwargs)
        else:
            return method(convert(X), **kwargs)

    def _call_robust(self, method, X, y, kwargs):
        if self._stored_conversion_approach is not None:
            approach = self._stored_conversion_approach
            self._log_attempt_using_stored_approach(approach)
            convert = approach.convert
            try:
                result = self._call_with_convert(method, convert, X, y, kwargs)
                self._log_success_using_stored_approach(approach)
                return result
            except Exception as e:
                self._log_failure_using_stored_approach(approach, e)
                raise
        else:
            for approach in DelegatingRobustTransformer.CONVERSION_APPROACHES:
                try:
                    self._log_attempt(approach)
                    result = self._call_with_convert(
                        method, approach.convert, X, y, kwargs)
                    self._log_success(approach)
                    self._stored_conversion_approach = approach
                    return result
                except approach.caught as e:
                    self._log_catch(approach, e)
                    continue
                except Exception as e:
                    self._log_error(approach, e)
                    raise

            self._log_failure_no_more_approaches()
            raise UnsuccessfulInputConversionError

    def _log_attempt_using_stored_approach(self, approach):
        logger.debug(
            'Attempting to convert using stored, previously-successful '
            'approach {approach.name!r}'
            .format(approach=approach))

    def _log_failure_using_stored_approach(self, approach, e):
        pretty_tb = self._get_pretty_tb()
        exc_name = type(e).__name__
        logger.debug(
            'Conversion unexpectedly failed using stored, '
            'previously-successful approach {approach.name!r} '
            'because of error {exc_name!r}\n\n{tb}'
            .format(approach=approach, exc_name=exc_name, tb=pretty_tb))

    def _log_success_using_stored_approach(self, approach):
        logger.debug(
            'Conversion with stored, previously-successful approach '
            '{approach.name!r} succeeded!'
            .format(approach=approach))

    def _log_attempt(self, approach):
        logger.debug(
            'Attempting to convert using approach {approach.name!r}...'
            .format(approach=approach))

    def _get_pretty_tb(self):
        tb = traceback.format_exc()
        pretty_tb = indent(tb, n=8)
        return pretty_tb

    def _log_catch(self, approach, e):
        pretty_tb = self._get_pretty_tb()
        exc_name = type(e).__name__
        logger.debug(
            'Conversion approach {approach.name!r} didn\'t work, '
            'caught exception {exc_name!r}\n\n{tb}'
            .format(approach=approach, exc_name=exc_name, tb=pretty_tb))

    def _log_error(self, approach, e):
        pretty_tb = self._get_pretty_tb()
        exc_name = type(e).__name__
        logger.debug(
            'Conversion failed during {approach.name!r} because of an '
            'unrecoverable error {exc_name!r}\n\n{tb}'
            .format(approach=approach, exc_name=exc_name, tb=pretty_tb))

    def _log_success(self, approach):
        logger.debug(
            'Conversion approach {approach.name!r} succeeded!'
            .format(approach=approach))

    def _log_failure_no_more_approaches(self):
        logger.info('Conversion failed, and we\'re not sure why...')


class Feature:
    def __init__(self, input, transformer, name=None, description=None,
                 output=None, source=None, options=None):
        self.input = input

        if is_seqcont(transformer):
            self.transformer = make_robust_transformer_pipeline(*transformer)
        else:
            self.transformer = DelegatingRobustTransformer(transformer)

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
        return '{clsname}({attrs_str})'.format(
            clsname=type(self).__name__, attrs_str=attrs_str)

    def as_input_transformer_tuple(self):
        return self.input, self.transformer

    def as_dataframe_mapper(self):
        return DataFrameMapper([
            self.as_input_transformer_tuple()
        ], input_df=True)
