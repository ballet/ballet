import os
import pickle

import h5py
import numpy as np
import pandas as pd

from ballet.compat import pathlib
from ballet.util.fs import splitext2
from ballet.util.log import logger

__all__ = ['read_tabular', 'write_tabular']


def _check_ext(ext, expected):
    if ext != expected:
        msg = ('File path has wrong extension: {} (expected {})'
               .format(ext, expected))
        raise ValueError(msg)


def write_tabular(obj, filepath):
    """Write tabular object in HDF5 or pickle format

    Args:
        obj (array or DataFrame): tabular object to write
        filepath (path-like): path to write to; must end in '.h5' or '.pkl'
    """
    _, fn, ext = splitext2(filepath)
    if ext == '.h5':
        _write_tabular_h5(obj, filepath)
    elif ext == '.pkl':
        _write_tabular_pickle(obj, filepath)
    else:
        raise NotImplementedError


def _write_tabular_pickle(obj, filepath):
    _, fn, ext = splitext2(filepath)
    _check_ext(ext, '.pkl')
    if isinstance(obj, np.ndarray):
        with open(str(filepath), 'wb') as f:
            pickle.dump(obj, f)
    elif isinstance(obj, pd.core.frame.NDFrame):
        obj.to_pickle(filepath)
    else:
        raise NotImplementedError


def _write_tabular_h5(obj, filepath):
    _, fn, ext = splitext2(filepath)
    _check_ext(ext, '.h5')
    if isinstance(obj, np.ndarray):
        with h5py.File(filepath, 'w') as hf:
            hf.create_dataset(fn, data=obj)
    elif isinstance(obj, pd.core.frame.NDFrame):
        obj.to_hdf(filepath, key=fn)
    else:
        raise NotImplementedError


def read_tabular(filepath):
    """Read tabular object in HDF5 or pickle format

    Args:
        filepath (path-like): path to read to; must end in '.h5' or '.pkl'
    """
    _, fn, ext = splitext2(filepath)
    if ext == '.h5':
        return _read_tabular_h5(filepath)
    elif ext == '.pkl':
        return _read_tabular_pickle(filepath)
    else:
        raise NotImplementedError


def _read_tabular_h5(filepath):
    _, fn, ext = splitext2(filepath)
    _check_ext(ext, '.h5')
    with h5py.File(filepath, 'r') as hf:
        dataset = hf[fn]
        data = dataset[:]
        return data


def _read_tabular_pickle(filepath):
    _, fn, ext = splitext2(filepath)
    _check_ext(ext, '.pkl')
    with open(str(filepath), 'rb') as f:
        return pickle.load(f)


def save_model(model, output_dir):
    logger.info('Saving model...')
    os.makedirs(output_dir, exist_ok=True)
    filepath = pathlib.Path(output_dir).joinpath('model.pkl')
    model.dump(filepath)
    logger.info('Saving model...DONE ({})'.format(filepath))


def save_predictions(y, output_dir):
    logger.info('Saving predictions...')
    os.makedirs(output_dir, exist_ok=True)
    filepath = pathlib.Path(output_dir).joinpath('predictions.pkl')
    write_tabular(y, filepath)
    logger.info('Saving predictions...DONE ({})'.format(filepath))
