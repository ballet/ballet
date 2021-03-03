import numpy as np
import pandas as pd
import pandas.util.testing as pdt
import pytest

from ballet.util.testing import (
    assert_array_equal, assert_array_not_equal, assert_frame_equal,
    assert_frame_not_equal, assert_index_equal, assert_index_not_equal,
    assert_series_equal, assert_series_not_equal,)


def test_assert_array_equal():
    a = np.arange(10 * 7).reshape(10, 7)
    b = a.copy()
    assert_array_equal(a, b)

    c = np.arange(9).reshape(3, 3)
    d = c + 1

    with pytest.raises(AssertionError):
        assert_array_equal(c, d)


def test_assert_array_not_equal():
    a = np.arange(10 * 7).reshape(10, 7)
    b = a.copy()

    with pytest.raises(AssertionError):
        assert_array_not_equal(a, b)

    c = np.arange(9).reshape(3, 3)
    d = c + 1

    assert_array_not_equal(c, d)


def test_assert_frame_equal():
    a = pdt.makeCustomDataframe(
        10, 7, data_gen_f=lambda row, col: row * col)
    b = a.copy()
    assert_frame_equal(a, b)

    c = a + 1
    with pytest.raises(AssertionError):
        assert_frame_equal(a, c)

    d = pdt.makeCustomDataframe(11, 9)
    with pytest.raises(AssertionError):
        assert_frame_equal(a, d)

    e = pdt.makeTimeSeries()
    with pytest.raises(AssertionError):
        assert_frame_equal(a, e)

    f = pd.DataFrame([1, 2, 3, 4])
    g = pd.Series([1, 2, 3, 4])
    with pytest.raises(AssertionError):
        assert_frame_equal(f, g)


def test_assert_frame_not_equal():
    a = pdt.makeCustomDataframe(
        10, 7, data_gen_f=lambda row, col: row * col)
    b = a.copy()
    with pytest.raises(AssertionError):
        assert_frame_not_equal(a, b)

    c = a + 1
    assert_frame_not_equal(a, c)

    d = pdt.makeCustomDataframe(11, 9)
    assert_frame_not_equal(a, d)

    e = pdt.makeTimeSeries()
    assert_frame_not_equal(a, e)

    f = pd.DataFrame([1, 2, 3, 4])
    g = pd.Series([1, 2, 3, 4])
    assert_frame_not_equal(f, g)


def test_assert_series_equal():
    a = pd.Series(np.arange(21))
    b = a.copy()
    assert_series_equal(a, b)

    c = a + 1
    with pytest.raises(AssertionError):
        assert_series_equal(a, c)

    d = pd.Series(np.arange(17))
    with pytest.raises(AssertionError):
        assert_series_equal(a, d)

    e = pdt.makeDataFrame()
    with pytest.raises(AssertionError):
        assert_series_equal(a, e)


def test_assert_series_not_equal():
    a = pd.Series(np.arange(21))
    b = a.copy()
    with pytest.raises(AssertionError):
        assert_series_not_equal(a, b)

    c = a + 1
    assert_series_not_equal(a, c)

    d = pd.Series(np.arange(17))
    assert_series_not_equal(a, d)

    e = pdt.makeDataFrame()
    assert_series_not_equal(a, e)


def test_assert_index_equal():
    a = pd.Index(np.arange(21))
    b = a.copy()
    assert_index_equal(a, b)

    c = a + 1
    with pytest.raises(AssertionError):
        assert_index_equal(a, c)

    d = pd.Index(np.arange(17))
    with pytest.raises(AssertionError):
        assert_index_equal(a, d)

    e = pdt.makeDataFrame()
    with pytest.raises(AssertionError):
        assert_index_equal(a, e)


def test_assert_index_not_equal():
    a = pd.Index(np.arange(21))
    b = a.copy()
    with pytest.raises(AssertionError):
        assert_index_not_equal(a, b)

    c = a + 1
    assert_index_not_equal(a, c)

    d = pd.Index(np.arange(17))
    assert_index_not_equal(a, d)

    e = pdt.makeDataFrame()
    assert_index_not_equal(a, e)
