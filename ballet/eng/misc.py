import funcy
import numpy as np
import pandas as pd

from ballet.eng.base import BaseTransformer, SimpleFunctionTransformer
from ballet.util import get_arr_desc

__all__ = ['IdentityTransformer', 'ValueReplacer', 'NamedFramer']


class IdentityTransformer(SimpleFunctionTransformer):
    def __init__(self):
        super().__init__(funcy.identity)
    
    def __eq__(self, other):
        if not isinstance(other, IdentityTransformer):
            return False
        return True


class ValueReplacer(BaseTransformer):
    def __init__(self, value, replacement):
        super().__init__()
        self.value = value
        self.replacement = replacement

    def __eq__(self, other):
        if not isinstance(other, ValueReplacer):
            return False
        if self.value != other.value:
            return False
        if self.replacement != other.replacement:
            return False
        return True

    def transform(self, X, **transform_kwargs):
        X = X.copy()
        mask = X == self.value
        X[mask] = self.replacement
        return X


class NamedFramer(BaseTransformer):
    def __init__(self, name):
        super().__init__()
        self.name = name
    
    def __eq__(self, other):
        if not isinstance(other, NamedFramer):
            return False
        if self.name != other.name:
            return False
        return True

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
