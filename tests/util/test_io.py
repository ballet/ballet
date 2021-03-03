import os
from unittest.mock import ANY, mock_open, patch

import numpy as np
import pandas as pd
import pytest

from ballet.util.io import (
    _check_ext, _write_tabular_h5, _write_tabular_pickle, write_tabular,)


@pytest.fixture
def array():
    return np.arange(10).reshape(2, 5)


@pytest.fixture
def frame():
    return pd.util.testing.makeDataFrame()


def test_check_ext_valid():
    ext = '.py'
    expected = ext
    _check_ext(ext, expected)


def test_check_ext_invalid_throws():
    ext = '.py'
    expected = '.txt'
    with pytest.raises(ValueError):
        _check_ext(ext, expected)


@patch('ballet.util.io._write_tabular_pickle')
@patch('ballet.util.io._write_tabular_h5')
def test_write_tabular(mock_write_tabular_h5,
                       mock_write_tabular_pickle):
    obj = object()
    filepath = '/foo/bar/baz.h5'
    write_tabular(obj, filepath)
    mock_write_tabular_h5.assert_called_once_with(obj, filepath)

    obj = object()
    filepath = '/foo/bar/baz.pkl'
    write_tabular(obj, filepath)
    mock_write_tabular_pickle.assert_called_once_with(obj, filepath)

    obj = object()
    filepath = '/foo/bar/baz.xyz'
    with pytest.raises(NotImplementedError):
        write_tabular(obj, filepath)


@patch('builtins.open', new_callable=mock_open)
@patch('pickle.dump')
def test_write_tabular_pickle_ndarray(mock_dump, mock_open, array):
    obj = array
    filepath = '/foo/bar/baz.pkl'
    _write_tabular_pickle(obj, filepath)
    mock_dump.assert_called_with(obj, ANY)


@patch('builtins.open', new_callable=mock_open)
def test_write_tabular_pickle_ndframe(mock_open, frame):
    obj = frame
    filepath = '/foo/bar/baz.pkl'

    with patch.object(obj, 'to_pickle') as mock_to_pickle:
        _write_tabular_pickle(obj, filepath)

    mock_to_pickle.assert_called_with(filepath)


def test_write_tabular_pickle_nonarray_raises():
    obj = object()
    filepath = '/foo/bar/baz.pkl'
    with pytest.raises(NotImplementedError):
        _write_tabular_pickle(obj, filepath)


def test_write_tabular_h5_ndarray(tmp_path, array):
    obj = array
    filepath = tmp_path.joinpath('baz.h5')
    _write_tabular_h5(obj, filepath)

    file_size = os.path.getsize(filepath)
    assert file_size > 0


def test_write_tabular_h5_ndframe(frame):
    obj = frame
    filepath = '/foo/bar/baz.h5'

    with patch.object(obj, 'to_hdf') as mock_to_hdf:
        _write_tabular_h5(obj, filepath)

    mock_to_hdf.assert_called_with(filepath, key=ANY)


@pytest.mark.xfail
def test_read_tabular():
    raise NotImplementedError


@pytest.mark.xfail
def test_save_model():
    raise NotImplementedError


@pytest.mark.xfail
def test_save_predictions():
    raise NotImplementedError
