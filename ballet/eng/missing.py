import numpy as np

from ballet.eng.base import BaseTransformer, GroupedFunctionTransformer

__all__ = ['LagImputer', 'NullFiller', 'NullIndicator']


class LagImputer(GroupedFunctionTransformer):
    """Fill missing values using group-specific lags"""

    def __init__(self, groupby_kwargs=None):
        super().__init__(lambda x: x.fillna(method='ffill'),
                         groupby_kwargs=groupby_kwargs)


class NullFiller(BaseTransformer):
    """Fill values passing a filter with a given replacement

    Args:
        isnull (callable): vectorized test of whether a value is consider null.
            Defaults to ``np.isnan``.
        replacement: replacement for each null value
    """

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
    """Indicate whether values are null or not"""

    def transform(self, X, **tranform_kwargs):
        return np.isnan(X).astype(int)
