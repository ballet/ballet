import copy
import sys

import numpy as np
import pandas as pd
import pytest

from ballet.util import (
    DeepcopyMixin, asarray2d, dfilter, falsy, get_arr_desc, has_nans, indent,
    load_sklearn_df, make_plural_suffix, nonnegative, quiet, truthy,)
from ballet.util.log import logger
from ballet.util.testing import assert_array_equal


def test_asarray2d_shape_n_x_2():
    # case: second dimension is > 1
    a = np.zeros((3, 2))
    result = asarray2d(a)
    assert_array_equal(result, a)


def test_asarray2d_shape_n_x_1():
    # case: second dimension == 1
    a = np.zeros((3, 1))
    result = asarray2d(a)
    assert_array_equal(result, a)


def test_asarray2d_shape_n():
    # case: second dimension not present
    a = np.zeros((3,))
    result = asarray2d(a)
    expected_shape = (3, 1)
    assert result.shape == expected_shape
    assert_array_equal(np.ravel(result), a)


def test_asarray2d_series():
    # case: pd.Series
    a = np.zeros((3,))
    ser = pd.Series(a)
    result = asarray2d(ser)
    assert result.shape[1] >= 1
    assert_array_equal(
        result, asarray2d(a)
    )


def test_asarray2d_df():
    # case: pd.DataFrame
    a = np.zeros((3, 2))
    df = pd.DataFrame(a)
    result = asarray2d(df)
    assert result.shape == df.shape
    assert result.shape[1] >= 1
    assert_array_equal(result, a)


def test_get_arr_desc_array():
    a = np.ones((2, 2))
    expected = 'ndarray (2, 2)'
    actual = get_arr_desc(a)
    assert actual == expected


def test_get_arr_desc_frame():
    df = pd.DataFrame()
    expected = 'DataFrame (0, 0)'
    actual = get_arr_desc(df)
    assert actual == expected


def test_get_arr_desc_object():
    obj = object()
    expected = 'object <no shape>'
    actual = get_arr_desc(obj)
    assert actual == expected


def test_indent():
    text = (
        'Hello\n'
        '  world\n'
        '...hi'
    )
    expected = (
        '    Hello\n'
        '      world\n'
        '    ...hi'
    )
    actual = indent(text, n=4)
    assert actual == expected


def test_make_plural_suffix_plural():
    objs = ['honda', 'toyota']
    suffix = make_plural_suffix(objs)
    actual = f'car{suffix}'
    expected = 'cars'
    assert actual == expected


def test_make_plural_suffix_singular():
    objs = ['honda']
    suffix = make_plural_suffix(objs)
    actual = f'car{suffix}'
    expected = 'car'
    assert actual == expected


def test_has_nans():
    objs_with_nans = [
        pd.DataFrame(data={'x': [1, np.nan], 'y': [np.nan, 2]}),
        pd.DataFrame(data={'x': [1, np.nan]}),
        pd.Series(data=[1, np.nan]),
        np.array([[1, np.nan], [np.nan, 2]]),
        np.array([[1, np.nan]]),
        np.array([1, np.nan]),
        np.array([1, np.nan]).T,
        np.array(np.nan),
    ]

    objs_without_nans = [
        pd.DataFrame(data={'x': [1, 0], 'y': [0, 2]}),
        pd.DataFrame(data={'x': [1, 0]}),
        pd.Series(data=[1, 0]),
        np.array([[1, 0], [0, 2]]),
        np.array([[1, 0]]),
        np.array([1, 0]),
        np.array([1, 0]).T,
        np.array(0),
    ]

    for obj in objs_with_nans:
        assert has_nans(obj)

    for obj in objs_without_nans:
        assert not has_nans(obj)


def test_dfilter():
    @dfilter(lambda x: x >= 0)
    def numbers():
        return [-1, 2, 0, -2]

    actual = numbers()
    expected = [2, 0]
    assert actual == expected


def test_load_sklearn_df():
    name = 'iris'
    X_df, y_df = load_sklearn_df(name)

    # validation on X_df
    assert X_df is not None
    assert isinstance(X_df, pd.DataFrame)

    # validation on y_df
    assert y_df is not None
    assert isinstance(y_df, pd.Series)


def test_quiet_stderr():
    @quiet
    def f():
        print('bar', file=sys.stderr)

    # doesn't check that nothing is printed to sys.stderr, but makes sure
    # the stream is reset properly (esp on py<3.5)
    stderr = sys.stderr
    f()
    assert sys.stderr is stderr


def test_deepcopy_mixin():
    class E(Exception):
        pass

    class A:
        def __init__(self, a):
            self.a = a

        def __deepcopy__(self, memo):
            raise E

    class B(DeepcopyMixin, A):
        pass

    a = A(1)
    with pytest.raises(E):
        copy.deepcopy(a)

    b = B(1)
    copy.deepcopy(b)


def test_falsy():
    matrix = (
        (False, True),
        (True, False),
        ('false', True),  # i.e., is falsy
        ('', True),
        ('true', False),  # i.e, is not falsy
        ('123', False),
        (73, False),
    )
    for input, expected in matrix:
        actual = falsy(input)
        assert expected == actual

        # truthy is complement of falsy
        actual = truthy(input)
        assert expected != actual


def test_nonnegative_positive_output():
    @nonnegative()
    def func():
        return 1

    assert 1 == func()


def test_nonnegative_negative_output():
    @nonnegative(name="Result")
    def func():
        return -1

    assert 0 == func()


def test_nonnegative_negative_introspection(caplog):
    @nonnegative()
    def estimate_something():
        return -1

    with caplog.at_level('WARNING', logger=logger.name):
        estimate_something()

    assert "Something" in caplog.text
