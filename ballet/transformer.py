import traceback
from collections import Counter
from inspect import signature
from typing import (
    Callable, Collection, List, NamedTuple, Sequence, Tuple, Union, cast)

import numpy as np
import pandas as pd
from funcy import identity, is_seqcont, select_values
from sklearn.base import BaseEstimator
from sklearn_pandas.pipeline import TransformerPipeline

from ballet.eng.base import BaseTransformer, SimpleFunctionTransformer
from ballet.exc import UnsuccessfulInputConversionError
from ballet.util import DeepcopyMixin, asarray2d, indent, quiet
from ballet.util.log import logger
from ballet.util.typing import OneOrMore, TransformerLike

RobustTransformer = Union[TransformerPipeline, 'DelegatingRobustTransformer']


def make_robust_transformer(
    transformer: OneOrMore[TransformerLike]
) -> RobustTransformer:
    if is_seqcont(transformer):
        transformer = cast(Collection[TransformerLike], transformer)
        transformers = list(
            map(_replace_callable_with_transformer, transformer))
        for t in transformers:
            _validate_transformer_api(t)
        return make_robust_transformer_pipeline(transformers)
    else:
        transformer = cast(TransformerLike, transformer)
        transformer = _replace_callable_with_transformer(transformer)
        _validate_transformer_api(transformer)
        return DelegatingRobustTransformer(transformer)


def _name_estimators(
    estimators: Sequence[BaseEstimator]
) -> List[Tuple[str, BaseEstimator]]:
    """Generate names for estimators.

    Adapted from sklearn.pipeline._name_estimators
    """

    def get_name(estimator):
        if isinstance(estimator, DelegatingRobustTransformer):
            return get_name(estimator._transformer)

        return type(estimator).__name__.lower()

    names = list(map(get_name, estimators))
    counter = dict(Counter(names))
    counter = select_values(lambda x: x > 1, counter)

    for i in reversed(range(len(estimators))):
        name = names[i]
        if name in counter:
            names[i] += "-%d" % counter[name]
            counter[name] -= 1

    return list(zip(names, estimators))


def make_transformer_pipeline(
    steps: Sequence[BaseTransformer],
) -> TransformerPipeline:
    """Construct a TransformerPipeline from the given estimators.

    Source: sklearn_pandas.cont_method
    """
    return TransformerPipeline(_name_estimators(steps))


def make_robust_transformer_pipeline(
    steps: Collection[BaseTransformer]
) -> TransformerPipeline:
    """Construct a transformer pipeline of DelegatingRobustTransformers"""
    return make_transformer_pipeline([
        DelegatingRobustTransformer(step) for step in steps
    ])


class ConversionApproach(NamedTuple):
    name: str
    convert: Callable
    caught: Collection[type]


class DelegatingRobustTransformer(DeepcopyMixin, BaseTransformer):
    """Robust transformer that delegates to underlying transformer

    This transformer is robust against different typed and shaped input data.
    It tries a variety of input data conversion approaches and passes the
    result to the underlying transformer, using the first approach that works.

    Args:
        transformer: a transformer object with fit and transform methods

    Raises:
        UnsuccessfulInputConversionError: If none of the conversion approaches
            work.

    """

    DEFAULT_CAUGHT = (ValueError, TypeError)

    CONVERSION_APPROACHES = [
        ConversionApproach('identity', identity, DEFAULT_CAUGHT),
        ConversionApproach('series', pd.Series, DEFAULT_CAUGHT),
        ConversionApproach('dataframe', pd.DataFrame, DEFAULT_CAUGHT),
        ConversionApproach('array', np.asarray, DEFAULT_CAUGHT),
        ConversionApproach('asarray2d', asarray2d, ()),
    ]

    def __init__(self, transformer: BaseTransformer):
        self._transformer = transformer
        self._stored_conversion_approach = None

    def __getattr__(self, attr):
        if '_transformer' in self.__dict__:
            return getattr(self._transformer, attr)
        else:
            raise AttributeError

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self):
        name = type(self).__name__
        return '{name}({transformer!r})'.format(
            name=name, transformer=self._transformer)

    @property
    def _tname(self) -> str:
        return type(self._transformer).__name__

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

    @quiet
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
            '{tname}: '
            'Attempting to convert using stored, '
            'previously-successful approach {approach.name!r}'
            .format(tname=self._tname, approach=approach))

    def _log_failure_using_stored_approach(self, approach, e):
        pretty_tb = self._get_pretty_tb()
        exc_name = type(e).__name__
        logger.debug(
            '{tname}: '
            'Conversion unexpectedly failed using stored, '
            'previously-successful approach {approach.name!r} '
            'because of error {exc_name!r}\n\n{tb}'
            .format(tname=self._tname, approach=approach, exc_name=exc_name,
                    tb=pretty_tb))

    def _log_success_using_stored_approach(self, approach):
        logger.debug(
            '{tname}: '
            'Conversion with stored, previously-successful approach '
            '{approach.name!r} succeeded!'
            .format(tname=self._tname, approach=approach))

    def _log_attempt(self, approach):
        logger.debug(
            '{tname}: '
            'Attempting to convert using approach {approach.name!r}...'
            .format(tname=self._tname, approach=approach))

    def _get_pretty_tb(self):
        tb = traceback.format_exc()
        pretty_tb = indent(tb, n=8)
        return pretty_tb

    def _log_catch(self, approach, e):
        pretty_tb = self._get_pretty_tb()
        exc_name = type(e).__name__
        logger.debug(
            '{tname}: '
            'Conversion approach {approach.name!r} didn\'t work, '
            'caught exception {exc_name!r}\n\n{tb}'
            .format(tname=self._tname, approach=approach, exc_name=exc_name,
                    tb=pretty_tb))

    def _log_error(self, approach, e):
        pretty_tb = self._get_pretty_tb()
        exc_name = type(e).__name__
        logger.debug(
            '{tname}: '
            'Conversion failed during {approach.name!r} because of '
            'an unrecoverable error {exc_name!r}\n\n{tb}'
            .format(tname=self._tname, approach=approach, exc_name=exc_name,
                    tb=pretty_tb))

    def _log_success(self, approach):
        logger.debug(
            '{tname}: '
            'Conversion approach {approach.name!r} succeeded!'
            .format(tname=self._tname, approach=approach))

    def _log_failure_no_more_approaches(self):
        logger.info('Conversion failed, and we\'re not sure why...')


def _validate_transformer_api(transformer: BaseTransformer):
    if not all(
        hasattr(transformer, attr)
        for attr in ['fit', 'transform', 'fit_transform']
    ):
        raise ValueError('Transformer object missing required attribute')

    sig_fit = signature(transformer.fit)
    if '(X, y=None' not in str(sig_fit):
        raise ValueError(
            'Invalid signature for transformer.fit: {sig_fit}'
            .format(sig_fit=sig_fit))

    sig_transform = signature(transformer.transform)
    if '(X' not in str(sig_transform):
        raise ValueError(
            'Invalid signature for transformer.transform: {sig_transform}'
            .format(sig_transform=sig_transform))


def _replace_callable_with_transformer(
    transformer: TransformerLike,
) -> BaseTransformer:
    if callable(transformer) and not isinstance(transformer, type):
        return SimpleFunctionTransformer(transformer)
    else:
        transformer = cast(BaseTransformer, transformer)
        return transformer
