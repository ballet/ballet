import funcy
import numpy as np
import pandas as pd
from scipy.special import boxcox1p
from scipy.stats import skew
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import check_is_fitted

from ballet.eng.base import BaseTransformer, ConditionalTransformer
from ballet.util import get_arr_desc

__all__ = [
    'IdentityTransformer',
    'BoxCoxTransformer',
    'ValueReplacer',
    'NamedFramer',
]


class IdentityTransformer(FunctionTransformer):
    """Simple transformer that passes through its input"""

    def __init__(self):
        super().__init__(func=funcy.identity, inverse_func=funcy.identity,
                         validate=False, check_inverse=False)


class BoxCoxTransformer(ConditionalTransformer):
    """Conditionally apply the Box-Cox transformation

    In the fit stage, determines which variables (columns) have absolute skew
    above ``threshold``. In the transform stage, applies the Box-Cox
    transformation of 1+x to each variable selected previously.

    Args:
        threshold: skew threshold.
        lmbda (float, default=0.0): Power parameter of the Box-Cox transform.

    See also:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.boxcox1p.html
    """

    def __init__(self, threshold, lmbda=0):
        def condition(X):
            return abs(skew(X)) > threshold

        def transform(X):
            return boxcox1p(X, lmbda)

        super().__init__(condition, transform)


class ValueReplacer(BaseTransformer):
    """Replace instances of some value with some replacement

    Args:
        value: value to replace (checked by equality testing)
        replacement: replacement
    """

    def __init__(self, value, replacement):
        super().__init__()
        self.value = value
        self.replacement = replacement

    def transform(self, X, **transform_kwargs):
        X = X.copy()
        mask = X == self.value
        X[mask] = self.replacement
        return X


class NamedFramer(BaseTransformer):
    """Convert object to named 1d DataFrame

    If transformation is successful, the resulting object is a DataFrame with a
    ``name`` attribute as given.

    Args:
        name: name for resulting DataFrame
    """

    def __init__(self, name):
        super().__init__()
        self.name = name

    def transform(self, X, **transform_kwargs):
        msg = "Couldn't convert object {} to named 1d DataFrame."
        if isinstance(X, pd.Index):
            return X.to_series().to_frame(name=self.name)
        elif isinstance(X, pd.Series):
            return X.to_frame(name=self.name)
        elif isinstance(X, pd.DataFrame):
            if X.shape[1] == 1:
                X = X.copy()
                X.columns = [self.name]
                return X
            else:
                raise ValueError(msg.format(get_arr_desc(X)))
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                return pd.DataFrame(data=X.reshape(-1, 1), columns=[self.name])
            elif X.ndim == 2 and X.shape[1] == 1:
                return pd.DataFrame(data=X, columns=[self.name])
            else:
                raise ValueError(msg.format(get_arr_desc(X)))

        raise TypeError(msg.format(get_arr_desc(X)))


class NullTransformer(BaseTransformer):

    def transform(self, X, **transform_kwargs):
        n = np.size(X, 0)
        return np.empty((n, 0))


class ComputedValueTransformer(BaseTransformer):

    def __init__(self, func, pass_y=False):
        self.func = func
        self.pass_y = pass_y

    def fit(self, X, y=None, **fit_kwargs):
        if self.pass_y:
            self.value_ = self.func(X, y=y)
        else:
            self.value_ = self.func(X)
        self.dtype_ = np.dtype(type(self.value_))

    def transform(self, X, **transform_kwargs):
        check_is_fitted(self, ['value_', 'dtype_'])
        return np.full_like(X, self.value_, dtype=self.dtype)
