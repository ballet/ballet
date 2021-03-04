import os
import pathlib
import pickle
from typing import Union

import h5py
import numpy as np
import pandas as pd
from stacklog import stacklog

from ballet.util.fs import splitext2
from ballet.util.log import logger
from ballet.util.typing import Pathy


def _check_ext(ext: str, expected: str):
    if ext != expected:
        raise ValueError(
            f'File path has wrong extension: {ext} (expected {expected})')


def write_tabular(obj: Union[np.ndarray, pd.DataFrame], filepath: Pathy):
    """Write tabular object in HDF5 or pickle format

    Args:
        obj: tabular object to write
        filepath: path to write to; must end in '.h5' or '.pkl'
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
        with open(filepath, 'wb') as f:
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


def read_tabular(filepath: Pathy):
    """Read tabular object in HDF5 or pickle format

    Args:
        filepath: path to read to; must end in '.h5' or '.pkl'
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
    with open(filepath, 'rb') as f:
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
    fn = pathlib.Path(output_dir).joinpath(f'{name}.pkl')
    with stacklog(logger.info, f'Saving {name} to {fn}'):
        os.makedirs(output_dir, exist_ok=True)
        savefn(thing, fn)


def load_table_from_config(input_dir: Pathy, config: dict) -> pd.DataFrame:
    """Load table from table config dict

    Args:
        input_dir: directory containing input files
        config: mapping with keys 'name', 'path', and 'pd_read_kwargs'.
    """
    path = pathlib.Path(input_dir).joinpath(config['path'])
    kwargs = config['pd_read_kwargs']
    return pd.read_csv(path, **kwargs)
