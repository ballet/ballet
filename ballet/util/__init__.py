from copy import deepcopy

import numpy as np

RANDOM_STATE = 1754


def asarray2d(a):
    """Cast to 2d array"""
    arr = np.asarray(a)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def get_arr_desc(arr):
    """Get array description, in the form '<array type> <array shape>'"""
    desc = '{typ} {shp}'
    typ = type(arr)
    shp = getattr(arr, 'shape', None)
    return desc.format(typ=typ, shp=shp)


def get_enum_keys(cls):
    return [attr for attr in dir(cls) if not attr.startswith('_')]


def get_enum_values(cls):
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


class DeepcopyMixin:

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result
