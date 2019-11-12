import pathlib
import warnings
from copy import deepcopy
from enum import Enum
from os import devnull

import numpy as np
import pandas as pd
import sklearn.datasets
from funcy import complement, decorator, lfilter, wraps

from ballet.compat import redirect_stderr, redirect_stdout
from ballet.exc import BalletWarning

RANDOM_STATE = 1754


def asarray2d(a):
    """Cast to 2d array"""
    arr = np.asarray(a)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def get_arr_desc(arr):
    """Get array description, in the form '<array type> <array shape>'"""
    type_ = type(arr).__name__  # see also __qualname__
    shape = getattr(arr, 'shape', None)
    if shape is not None:
        desc = '{type_} {shape}'
    else:
        desc = '{type_} <no shape>'
    return desc.format(type_=type_, shape=shape)


def get_enum_keys(cls):
    return [attr for attr in dir(cls) if not attr.startswith('_')]


def get_enum_values(cls):
    if issubclass(cls, Enum):
        return [getattr(cls, attr).value for attr in get_enum_keys(cls)]
    else:
        return [getattr(cls, attr) for attr in get_enum_keys(cls)]


def indent(text, n=4):
    """Indent each line of text by n spaces"""
    _indent = ' ' * n
    return '\n'.join(_indent + line for line in text.split('\n'))


def make_plural_suffix(obj, suffix='s'):
    if len(obj) != 1:
        return suffix
    else:
        return ''


@decorator
def whether_failures(call):
    """Collects failures and return (success, list_of_failures)"""
    failures = list(call())
    return not failures, failures


def has_nans(obj):
    """Check if obj has any NaNs

    Compatible with different behavior of np.isnan, which sometimes applies
    over all axes (py35, py35) and sometimes does not (py34).
    """
    nans = np.isnan(obj)
    while np.ndim(nans):
        nans = np.any(nans)
    return bool(nans)


@decorator
def dfilter(call, pred):
    """Decorate a callable with a filter that accepts a predicate

    Example::

        >>> @dfilter(lambda x: x >= 0)
        ... def numbers():
        ...     return [-1, 2, 0, -2]
        [2, 0]
    """
    return lfilter(pred, call())


def load_sklearn_df(name):
    method_name = 'load_{name}'.format(name=name)
    method = getattr(sklearn.datasets, method_name)
    data = method()
    X_df = pd.DataFrame(data=data.data, columns=data.feature_names)
    y_df = pd.Series(data.target, name='target')
    return X_df, y_df


@decorator
def quiet(call):
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull), redirect_stdout(fnull):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return call()


class DeepcopyMixin:

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result


def one_or_raise(l):
    n = len(l)
    if n == 1:
        return l[0]
    else:
        raise ValueError('Expected exactly 1 element, but got {n}'
                         .format(n=n))


def needs_path(f):
    """Wraps a function that accepts path-like to give it a pathlib.Path"""
    @wraps(f)
    def wrapped(pathlike, *args, **kwargs):
        path = pathlib.Path(pathlike)
        return f(path, *args, **kwargs)
    return wrapped


def warn(msg):
    """Issue a warning message of category BalletWarning"""
    warnings.warn(msg, category=BalletWarning)


@decorator
def raiseifnone(call):
    """Decorate a function to raise a ValueError if result is None"""
    result = call()
    if result is None:
        raise ValueError
    else:
        return result


def falsy(o):
    return isinstance(o, str) and (o.lower() == 'false' or o == '')


truthy = complement(falsy)
