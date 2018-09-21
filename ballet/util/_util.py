import funcy
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

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


def assertion_method(func):
    '''Evaluate func, returning T if no errors and F if AssertionError'''
    @funcy.wraps(func)
    def wrapped(*args, **kwargs):
        try:
            func(*args, **kwargs)
            return True
        except AssertionError:
            return False
    wrapped.is_check = True
    return wrapped


class NoFitMixin:
    def fit(self, X, y=None, **fit_kwargs):
        return self


class SimpleFunctionTransformer(BaseEstimator, NoFitMixin, TransformerMixin):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def transform(self, X, **transform_kwargs):
        return self.func(X)


class IdentityTransformer(SimpleFunctionTransformer):
    def __init__(self):
        super().__init__(funcy.identity)
