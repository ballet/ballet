import random
from unittest.util import _common_shorten_repr

import funcy
import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.util.testing as pdt

EPSILON = 1e-4


@funcy.decorator
def withfailure(call, format):
    try:
        call()
    except AssertionError:
        standardMsg = format.format(
            *_common_shorten_repr(call.first, call.second))
        msg = call.self._formatMessage(call.msg, standardMsg)
        raise call.self.fail(msg)


class ArrayLikeEqualityTestingMixin:
    """Mix-in to enable comparisons of array-like objects in unit tests"""

    @withfailure('{} != {}')
    def assertArrayEqual(self, first, second, msg=None):
        """Test that arrays first and second are equal"""
        npt.assert_array_equal(first, second, verbose=False)

    @withfailure('{} == {}')
    def assertArrayNotEqual(self, first, second, msg=None):
        """Test that arrays first and second are not equal"""
        try:
            npt.assert_array_equal(first, second, verbose=False)
        except AssertionError:
            pass
        else:
            raise AssertionError

    @withfailure('{} !≈ {}')
    def assertArrayAlmostEqual(
            self, first, second, places=6, msg=None, delta=None):
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

    @withfailure('{} ≈ {}')
    def assertArrayNotAlmostEqual(
            self, first, second, places=6, msg=None, delta=None):
        """Test that arrays first and second are not almost equal"""

        if delta is not None:
            places = int(-np.log10(delta) + np.log10(1.5))

        try:
            npt.assert_array_almost_equal(
                first, second, decimal=places, verbose=False)
        except AssertionError:
            pass
        else:
            raise AssertionError

    def assertFrameEqual(self, first, second, msg=None, **kwargs):
        """Test that DataFrames first and second are equal"""
        self._assertPandasEqual(
            pdt.assert_frame_equal, first, second, msg=msg, **kwargs)

    def assertFrameNotEqual(self, first, second, msg=None, **kwargs):
        """Test that DataFrames first and second are not equal"""
        self._assertPandasNotEqual(
            pdt.assert_frame_equal, first, second, msg=msg, **kwargs)

    def assertSeriesEqual(self, first, second, msg=None, **kwargs):
        """Test that Series first and second are equal"""
        self._assertPandasEqual(
            pdt.assert_series_equal, first, second, msg=msg, **kwargs)

    def assertSeriesNotEqual(self, first, second, msg=None, **kwargs):
        """Test that Series first and second are not equal"""
        self._assertPandasNotEqual(
            pdt.assert_series_equal, first, second, msg=msg, **kwargs)

    def assertIndexEqual(self, first, second, msg=None, **kwargs):
        """Test that Index first and second are equal"""
        self._assertPandasEqual(
            pdt.assert_index_equal, first, second, msg=msg, **kwargs)

    def assertIndexNotEqual(self, first, second, msg=None, **kwargs):
        """Test that Index first and second are not equal"""
        self._assertPandasNotEqual(
            pdt.assert_index_equal, first, second, msg=msg, **kwargs)

    def assertPandasObjectEqual(self, first, second, msg=None, **kwargs):
        """Test that arbitrary Pandas objects first and second are equal"""
        is_pdobj = funcy.isa(pd.core.base.PandasObject)
        if is_pdobj(first) and is_pdobj(second):
            if isinstance(first, type(second)):
                if isinstance(first, pd.DataFrame):
                    self.assertFrameEqual(first, second, msg=msg, **kwargs)
                elif isinstance(first, pd.Series):
                    self.assertSeriesEqual(first, second, msg=msg, **kwargs)
                elif isinstance(first, pd.Index):
                    self.assertIndexEqual(first, second, msg=msg, **kwargs)
                else:
                    # unreachable?
                    raise AssertionError('you found a bug: unreachable code')
        else:
            standardMsg = '{} and {} are uncomparable types'.format(
                *_common_shorten_repr(first, second))
            msg = self._formatMessage(msg, standardMsg)
            raise self.failureException(msg)

    def assertPandasObjectNotEqual(self, first, second, msg=None, **kwargs):
        """Test that arbitrary Pandas objects first and second are not equal"""
        is_pdobj = funcy.isa(pd.core.base.PandasObject)
        if is_pdobj(first) and is_pdobj(second):
            if isinstance(first, type(second)):
                if isinstance(first, pd.DataFrame):
                    self.assertFrameNotEqual(first, second, msg=msg, **kwargs)
                elif isinstance(first, pd.Series):
                    self.assertSeriesNotEqual(first, second, msg=msg, **kwargs)
                elif isinstance(first, pd.Index):
                    self.assertIndexNotEqual(first, second, msg=msg, **kwargs)
                else:
                    # unreachable?
                    raise AssertionError('you found a bug: unreachable code')
        else:
            # it's great that they are uncomparable types :)
            pass

    @withfailure('{} != {}')
    def _assertPandasEqual(self, func, first, second, msg=None, **kwargs):
        func(first, second, **kwargs)

    @withfailure('{} == {}')
    def _assertPandasNotEqual(self, func, first, second, msg=None, **kwargs):
        try:
            func(first, second, **kwargs)
        except AssertionError:
            pass
        else:
            raise AssertionError


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
        logger.exception('Error was thrown using seed {}'.format(seed))
