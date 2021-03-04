import random
from contextlib import contextmanager
from unittest.util import _common_shorten_repr

import funcy
import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt

EPSILON = 1e-4


@funcy.contextmanager
def seeded(seed):
    """Set seed, run code, then restore rng state"""
    if seed is not None:
        np_random_state = np.random.get_state()
        random_state = random.getstate()
        np.random.seed(seed)
        random.seed(seed)

    yield

    if seed is not None:
        np.random.set_state(np_random_state)
        random.setstate(random_state)


@funcy.contextmanager
def log_seed_on_error(logger, seed=None):
    """Store seed, run code, and report seed if error"""
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    try:
        with seeded(seed):
            yield
    except Exception:
        logger.exception(f'Error was thrown using seed {seed}')


@contextmanager
def _invert_assertion():
    try:
        yield
    except AssertionError:
        pass
    else:
        raise AssertionError


def assert_array_equal(first, second):
    npt.assert_array_equal(first, second, verbose=False)


def assert_array_not_equal(first, second):
    with _invert_assertion():
        npt.assert_array_equal(first, second, verbose=False)


def assert_array_almost_equal(first, second, places=6, delta=None):
    """Test that arrays first and second are almost equal

    There are two ways to control the tolerance of the approximate
    equality test, using ``places`` or ``delta``. All values are compared
    element-wise.
    ``delta``
    is provided, the allowed tolerance between the values is set to
    ``delta. Else if ``places`` is provided, values are compared up to
    ``places`` places. Else, ``places`` defaults to 6.

    Args:
        first: first object
        second: second object
        places (int, default=6): elements are compared up to this many
            places
        delta (float): the allowed tolerance between elements
    """

    # convert between both "places" and "delta" allowed to just "decimal"
    # numpy says abs(desired-actual) < 1.5 * 10**(-decimal)
    # => delta = 1.5 * 10**(-decimal)
    # => delta / 1.5 = 10**(-decimal)
    # => log10(delta/1.5) = -decimal
    # => decimal = -log10(delta) + log10(1.5)
    if delta is not None:
        places = int(-np.log10(delta) + np.log10(1.5))

    npt.assert_array_almost_equal(
        first, second, decimal=places, verbose=False)


def assert_array_not_almost_equal(
        first, second, places=6, delta=None):
    """Test that arrays first and second are not almost equal"""

    if delta is not None:
        places = int(-np.log10(delta) + np.log10(1.5))

    with _invert_assertion():
        npt.assert_array_almost_equal(
            first, second, decimal=places, verbose=False)


def assert_frame_equal(first, second, **kwargs):
    """Test that DataFrames first and second are equal"""
    _assert_pandas_equal(
        pdt.assert_frame_equal, first, second, **kwargs)


def assert_frame_not_equal(first, second, **kwargs):
    """Test that DataFrames first and second are not equal"""
    _assert_pandas_not_equal(
        pdt.assert_frame_equal, first, second, **kwargs)


def assert_series_equal(first, second, **kwargs):
    """Test that Series first and second are equal"""
    _assert_pandas_equal(
        pdt.assert_series_equal, first, second, **kwargs)


def assert_series_not_equal(first, second, **kwargs):
    """Test that Series first and second are not equal"""
    _assert_pandas_not_equal(
        pdt.assert_series_equal, first, second, **kwargs)


def assert_index_equal(first, second, **kwargs):
    """Test that Index first and second are equal"""
    _assert_pandas_equal(
        pdt.assert_index_equal, first, second, **kwargs)


def assert_index_not_equal(first, second, **kwargs):
    """Test that Index first and second are not equal"""
    _assert_pandas_not_equal(
        pdt.assert_index_equal, first, second, **kwargs)


_is_pdobj = funcy.isa(pd.core.base.PandasObject)


def assert_pandas_object_equal(first, second, **kwargs):
    """Test that arbitrary Pandas objects first and second are equal"""
    if _is_pdobj(first) and _is_pdobj(second):
        if isinstance(first, type(second)):
            if isinstance(first, pd.DataFrame):
                assert_frame_equal(first, second, **kwargs)
            elif isinstance(first, pd.Series):
                assert_series_equal(first, second, **kwargs)
            elif isinstance(first, pd.Index):
                assert_index_equal(first, second, **kwargs)
            else:
                # unreachable?
                raise AssertionError('you found a bug: unreachable code')
    else:
        msg = '{} and {} are uncomparable types'.format(
            *_common_shorten_repr(first, second))
        raise AssertionError(msg)


def assert_pandas_object_not_equal(first, second, **kwargs):
    """Test that arbitrary Pandas objects first and second are not equal"""
    if _is_pdobj(first) and _is_pdobj(second):
        if isinstance(first, type(second)):
            if isinstance(first, pd.DataFrame):
                assert_frame_not_equal(first, second, **kwargs)
            elif isinstance(first, pd.Series):
                assert_series_not_equal(first, second, **kwargs)
            elif isinstance(first, pd.Index):
                assert_index_not_equal(first, second, **kwargs)
            else:
                # unreachable?
                raise AssertionError('you found a bug: unreachable code')
    else:
        # it's great that they are uncomparable types :)
        pass


def _assert_pandas_equal(func, first, second, **kwargs):
    func(first, second, **kwargs)


def _assert_pandas_not_equal(func, first, second, **kwargs):
    try:
        func(first, second, **kwargs)
    except AssertionError:
        pass
    else:
        raise AssertionError
