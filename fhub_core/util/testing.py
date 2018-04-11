from unittest.util import _common_shorten_repr

import funcy
import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.util.testing as pdt


class ArrayLikeEqualityTestingMixin:

    def assertArrayEqual(self, first, second, msg=None):
        try:
            npt.assert_array_equal(first, second, verbose=False)
        except AssertionError:
            standardMsg = '{} != {}'.format(
                _common_shorten_repr(first, second))
            msg = self._formatMessage(msg, standardMsg)
            raise self.failureException(msg)

    def assertArrayAlmostEqual(
            self, first, second, places=6, msg=None, delta=None):

        # convert between both "places" and "delta" allowed to just "decimal"
        # numpy says abs(desired-actual) < 1.5 * 10**(-decimal)
        # => delta = 1.5 * 10**(-decimal)
        # => delta / 1.5 = 10**(-decimal)
        # => log10(delta/1.5) = -decimal
        # => decimal = -log10(delta) + log10(1.5)
        if delta is not None:
            places = int(-np.log10(delta) + np.log10(1.5))

        try:
            npt.assert_array_almost_equal(
                first, second, decimal=places, verbose=False)
        except AssertionError:
            standardMsg = '{} != {}'.format(
                _common_shorten_repr(first, second))
            msg = self._formatMessage(msg, standardMsg)
            raise self.failureException(msg)

    def _assertPandasEqual(self, func, first, second, msg=None, **kwargs):
        try:
            func(first, second, **kwargs)
        except AssertionError:
            standardMsg = '{} != {}'.format(
                _common_shorten_repr(first, second))
            msg = self._formatMessage(msg, standardMsg)
            raise self.failureException(msg)

    def assertFrameEqual(self, first, second, msg=None, **kwargs):
        self._assertPandasEqual(
            pdt.assert_frame_equal, first, second, msg=None, **kwargs)

    def assertSeriesEqual(self, first, second, msg=None, **kwargs):
        self._assertPandasEqual(
            pdt.assert_series_equal, first, second, msg=None, **kwargs)

    def assertIndexEqual(self, first, second, msg=None, **kwargs):
        self._assertPandasEqual(
            pdt.assert_index_equal, first, second, msg=None, **kwargs)

    def assertPandasObjectEqual(self, first, second, msg=None, **kwargs):
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
                    pass
        else:
            standardMsg = '{} and {} are uncomparable types'.format(
                _common_shorten_repr(first, second))
            msg = self._formatMessage(msg, standardMsg)
            raise self.failureException(msg)
