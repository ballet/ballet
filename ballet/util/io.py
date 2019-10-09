import os
import pathlib
import pickle

import h5py
import numpy as np
import pandas as pd

from ballet.compat import safepath
from ballet.util.fs import splitext2
from ballet.util.log import logger, stacklog


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
        with open(safepath(filepath), 'wb') as f:
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
    with open(safepath(filepath), 'rb') as f:
        return pickle.load(f)


def save_model(model, output_dir, name='model'):
    _save_thing(model, output_dir, name,
                savefn=lambda thing, fn: thing.dump(fn))


def save_predictions(ypred, output_dir, name='predictions'):
    """Save predictions to output directory"""
    _save_thing(ypred, output_dir, name)


def save_features(X, output_dir, name='features'):
    """Save built features to output directory"""
    _save_thing(X, output_dir, name)


def save_targets(y, output_dir, name='target'):
    """Save built target to output directory"""
    _save_thing(y, output_dir, name)


def _save_thing(thing, output_dir, name, savefn=write_tabular):
    fn = pathlib.Path(output_dir).joinpath('{}.pkl'.format(name))
    with stacklog(logger.info, 'Saving {} to {}'.format(name, fn)):
        os.makedirs(output_dir, exist_ok=True)
        savefn(thing, fn)


def load_table_from_config(input_dir, config):
    """Load table from table config dict

    Args:
        input_dir (path-like): directory containing input files
        config (dict): mapping with keys 'name', 'path', and 'pd_read_kwargs'.

    Returns:
        pd.DataFrame
    """
    path = pathlib.Path(input_dir).joinpath(config['path'])
    kwargs = config['pd_read_kwargs']
    return pd.read_csv(path, **kwargs)
