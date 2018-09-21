import unittest

import numpy as np
import pandas as pd
import pandas.util.testing as pdt

from ballet.util.testing import ArrayLikeEqualityTestingMixin


class TestArrayLikeEqualityTestingMixin(
        ArrayLikeEqualityTestingMixin, unittest.TestCase):

    def setUp(self):
        pass

    def test_assert_array_equal(self):
        a = np.arange(10 * 7).reshape(10, 7)
        b = a.copy()
        self.assertArrayEqual(a, b)

        c = np.arange(9).reshape(3, 3)
        d = c + 1

        with self.assertRaises(AssertionError):
            self.assertArrayEqual(c, d)

    def test_assert_array_not_equal(self):
        a = np.arange(10 * 7).reshape(10, 7)
        b = a.copy()

        with self.assertRaises(AssertionError):
            self.assertArrayNotEqual(a, b)

        c = np.arange(9).reshape(3, 3)
        d = c + 1

        self.assertArrayNotEqual(c, d)

    def test_assert_frame_equal(self):
        a = pdt.makeCustomDataframe(
            10, 7, data_gen_f=lambda row, col: row * col)
        b = a.copy()
        self.assertFrameEqual(a, b)

        c = a + 1
        with self.assertRaises(
                AssertionError, msg='Should find that frames are same size '
                                    'but different values'):
            self.assertFrameEqual(a, c)

        d = pdt.makeCustomDataframe(11, 9)
        with self.assertRaises(
                AssertionError, msg='Should find that frames are different '
                                    'size'):
            self.assertFrameEqual(a, d)

        e = pdt.makeTimeSeries()
        with self.assertRaises(
                AssertionError, msg='Should find that the second arg is '
                                    'wrong type'):
            self.assertFrameEqual(a, e)

        f = pd.DataFrame([1, 2, 3, 4])
        g = pd.Series([1, 2, 3, 4])
        with self.assertRaises(
                AssertionError,
                msg='Should find that the second arg is wrong type, '
                    'even though they have the same values'):
            self.assertFrameEqual(f, g)

    def test_assert_frame_not_equal(self):
        a = pdt.makeCustomDataframe(
            10, 7, data_gen_f=lambda row, col: row * col)
        b = a.copy()
        with self.assertRaises(AssertionError,
                               msg='frames are actually equal'):
            self.assertFrameNotEqual(a, b)

        c = a + 1
        self.assertFrameNotEqual(a, c)

        d = pdt.makeCustomDataframe(11, 9)
        self.assertFrameNotEqual(a, d)

        e = pdt.makeTimeSeries()
        self.assertFrameNotEqual(a, e)

        f = pd.DataFrame([1, 2, 3, 4])
        g = pd.Series([1, 2, 3, 4])
        self.assertFrameNotEqual(f, g)

    def test_assert_series_equal(self):
        a = pd.Series(np.arange(21))
        b = a.copy()
        self.assertSeriesEqual(a, b)

        c = a + 1
        with self.assertRaises(
                AssertionError, msg='Should find that series are same size '
                                    'but different values'):
            self.assertSeriesEqual(a, c)

        d = pd.Series(np.arange(17))
        with self.assertRaises(
                AssertionError, msg='Should find that series are different '
                                    ' sizes'):
            self.assertSeriesEqual(a, d)

        e = pdt.makeDataFrame()
        with self.assertRaises(
                AssertionError, msg='Should find that the second arg is '
                                    'wrong type'):
            self.assertSeriesEqual(a, e)

    def test_assert_series_not_equal(self):
        a = pd.Series(np.arange(21))
        b = a.copy()
        with self.assertRaises(
                AssertionError, msg='Should find that the series are '
                                    'indeed equal'):
            self.assertSeriesNotEqual(a, b)

        c = a + 1
        self.assertSeriesNotEqual(a, c)

        d = pd.Series(np.arange(17))
        self.assertSeriesNotEqual(a, d)

        e = pdt.makeDataFrame()
        self.assertSeriesNotEqual(a, e)

    def test_assert_index_equal(self):
        a = pd.Index(np.arange(21))
        b = a.copy()
        self.assertIndexEqual(a, b)

        c = a + 1
        with self.assertRaises(
                AssertionError, msg='Should find that indexes are same '
                                    'size but different values'):
            self.assertIndexEqual(a, c)

        d = pd.Index(np.arange(17))
        with self.assertRaises(
                AssertionError, msg='Should find that indexes are '
                                    'different sizes'):
            self.assertIndexEqual(a, d)

        e = pdt.makeDataFrame()
        with self.assertRaises(
                AssertionError, msg='Should find that the second arg is '
                                    'wrong type'):
            self.assertIndexEqual(a, e)

    def test_assert_index_not_equal(self):
        a = pd.Index(np.arange(21))
        b = a.copy()
        with self.assertRaises(
                AssertionError, msg='Should find that indexes are '
                                    'indeed equal'):
            self.assertIndexNotEqual(a, b)

        c = a + 1
        self.assertIndexNotEqual(a, c)

        d = pd.Index(np.arange(17))
        self.assertIndexNotEqual(a, d)

        e = pdt.makeDataFrame()
        self.assertIndexNotEqual(a, e)
