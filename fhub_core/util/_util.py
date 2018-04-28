import pathlib
import pkgutil

import numpy as np

RANDOM_STATE = 1754


def asarray2d(a):
    arr = np.asarray(a)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def get_arr_desc(arr):
    desc = '{typ} {shp}'
    typ = type(arr)
    shp = getattr(arr, 'shape', None)
    return desc.format(typ=typ, shp=shp)


def indent(text, n=4):
    _indent = ' ' * n
    return '\n'.join([_indent + line for line in text.split('\n')])


def import_module_at_path(modname, modpath):
    '''Import module from path'''
    modpath = pathlib.Path(modpath)
    parentpath = str(modpath.parent)
    modpath = str(modpath)
    importer = pkgutil.get_importer(parentpath)
    mod = importer.find_module(modname).load_module(modname)
    return mod
