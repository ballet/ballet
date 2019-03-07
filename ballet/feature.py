import traceback
from collections import Counter, namedtuple
from inspect import signature

import numpy as np
import pandas as pd
from funcy import cached_property, identity, is_seqcont, iterable, \
    select_values
from sklearn.base import TransformerMixin
from sklearn_pandas import DataFrameMapper
from sklearn_pandas.pipeline import TransformerPipeline

from ballet.eng.base import SimpleFunctionTransformer
from ballet.eng.misc import NullTransformer
from ballet.exc import UnsuccessfulInputConversionError
from ballet.util import DeepcopyMixin, asarray2d, indent, quiet
from ballet.util.log import logger

__all__ = ['make_mapper', 'Feature']


def make_mapper(features):
    """Make a DataFrameMapper from a feature or list of features

    Args:
        features (Union[Feature, List[Feature]]): feature or list of features

    Returns:
        DataFrameMapper: mapper made from features
    """
    if not features:
        features = Feature(input=[], transformer=NullTransformer())
    if not iterable(features):
        features = (features, )
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
    counter = select_values(lambda x: x > 1, counter)

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
    """Construct a transformer pipeline of DelegatingRobustTransformers"""
    steps = list(map(DelegatingRobustTransformer, steps))
    return make_transformer_pipeline(*steps)


ConversionApproach = namedtuple('ConversionApproach', 'name convert caught')


class DelegatingRobustTransformer(DeepcopyMixin, TransformerMixin):
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

    def __init__(self, transformer):
        self._transformer = transformer
        self._stored_conversion_approach = None

    def __getattr__(self, attr):
        return getattr(self._transformer, attr)

    def __repr__(self):
        name = type(self).__name__
        return '{name}({transformer!r})'.format(
            name=name, transformer=self._transformer)

    @property
    def _tname(self):
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


def _validate_transformer_api(transformer):
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
            .format(sig_fit=sig_fit))


def _replace_callable_with_transformer(transformer):
    if callable(transformer) and not isinstance(transformer, type):
        return SimpleFunctionTransformer(transformer)
    else:
        return transformer


class Feature:
    """A logical feature

    Conceptually, a logical feature is a learned function that maps raw
    variables in one data instance to a vector of feature values. A logical
    feature can produce either a scalar feature value for each instance or a
    vector of feature values, as in the case of an embedding technique like PCA
    or the one-hot encoding of a categorical variable.

    Args:
        input (str, Collection[str]): required columns from the input
            dataframe needed for the transformation
        transformer (transformer-like, List[transformer-like]): transformer, or
            sequence of transformers. A "transformer" is an instance of a class
            that provides a fit/transform-style learned transformation.
            Alternately, a callable can be provided, either by itself or in a
            list, in which case it will be converted into a
            ``SimpleFunctionTransformer`` for convenience.
        name (str, optional): name of the feature
        description (str, optional): description of the feature
        output (str, list[str]): ordered sequence of names of features
            produced by this transformer
        source (path-like): the source file in which this feature was defined
        options (dict): options
    """

    def __init__(self, input, transformer, name=None, description=None,
                 output=None, source=None, options=None):
        self.input = input

        if is_seqcont(transformer):
            transformer = map(_replace_callable_with_transformer, transformer)
            map(_validate_transformer_api, transformer)
            self.transformer = make_robust_transformer_pipeline(*transformer)
        else:
            transformer = _replace_callable_with_transformer(transformer)
            _validate_transformer_api(transformer)
            self.transformer = DelegatingRobustTransformer(transformer)

        self.name = name
        self.description = description
        self.output = output  # unused
        self.source = source
        self.options = options if options is not None else {}

    @cached_property
    def _init_attr_list(self):
        return list(signature(self.__init__).parameters)

    def __repr__(self):
        attr_list = self._init_attr_list
        attrs_str = ', '.join(
            '{attr_name}={attr_val!r}'.format(
                attr_name=attr, attr_val=getattr(self, attr)
            ) for attr in attr_list
        )
        return '{clsname}({attrs_str})'.format(
            clsname=type(self).__name__, attrs_str=attrs_str)

    def as_input_transformer_tuple(self):
        """Return an tuple for passing to DataFrameMapper constructor"""
        return (self.input, self.transformer, {'alias': self.output})

    def as_dataframe_mapper(self):
        """Return standalone DataFrameMapper with this feature"""
        return DataFrameMapper([
            self.as_input_transformer_tuple()
        ], input_df=True)
