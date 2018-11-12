import funcy
import numpy as np
import pandas as pd
from scipy.special import boxcox1p
from scipy.stats import skew

from ballet.eng.base import BaseTransformer, SimpleFunctionTransformer
from ballet.util import get_arr_desc

__all__ = ['IdentityTransformer', 'BoxCoxTransformer' 'ValueReplacer', 'NamedFramer']


class IdentityTransformer(SimpleFunctionTransformer):
    def __init__(self):
        super().__init__(funcy.identity)


class BoxCoxTransformer(BaseTransformer):
    def __init__(self, threshold, lmbda=0):
        super().__init__()
        self.threshold = threshold
        self.lmbda = lmbda

    def fit(self, X, y=None, **fit_args):
        self.features_to_transform_ = skew(X) > self.threshold

    def transform(self, X, **transform_args):
        return boxcox1p(X[self.features_to_transform_], self.lmbda)



class ValueReplacer(BaseTransformer):
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
