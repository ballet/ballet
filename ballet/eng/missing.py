import numpy as np

from ballet.eng.base import BaseTransformer, GroupedFunctionTransformer

__all__ = ['LagImputer', 'NullFiller', 'NullIndicator']


class LagImputer(GroupedFunctionTransformer):
    def __init__(self, groupby_kwargs=None):
        super().__init__(lambda x: x.fillna(method='ffill'),
                         groupby_kwargs=groupby_kwargs)


class NullFiller(BaseTransformer):
    def __init__(self, isnull=None, replacement=0.0):
        super().__init__()
        if isnull is None:
            self.isnull = np.isnan
        else:
            self.isnull = isnull
        self.replacement = replacement

    def transform(self, X, **transform_kwargs):
        X = X.copy()
        mask = self.isnull(X)
        X[mask] = self.replacement
        return X


class NullIndicator(BaseTransformer):
    def transform(self, X, **tranform_kwargs):
        return np.isnan(X).astype(int)
