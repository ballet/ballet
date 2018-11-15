import funcy
import numpy as np
import pandas as pd
from scipy.special import boxcox1p
from scipy.stats import skew
from sklearn.utils.validation import check_is_fitted

from ballet.eng.base import BaseTransformer, SimpleFunctionTransformer
from ballet.util import get_arr_desc

__all__ = ['IdentityTransformer', 'BoxCoxTransformer', 'ValueReplacer', 'NamedFramer']


class IdentityTransformer(SimpleFunctionTransformer):
    def __init__(self):
        super().__init__(funcy.identity)


class BoxCoxTransformer(BaseTransformer):
    def __init__(self, threshold, lmbda=0):
        super().__init__()
        self.threshold = threshold
        self.lmbda = lmbda

    def fit(self, X, y=None, **fit_args):
        self.features_to_transform_ = abs(skew(X)) > self.threshold
        if (isinstance(X, pd.DataFrame)):
            # Hack to get a mask over columns
            self.features_to_transform_ = X.columns.putmask(~self.features_to_transform_, None) & X.columns
        return self

    def transform(self, X, **transform_args):
        check_is_fitted(self, 'features_to_transform_')
        if isinstance(X, pd.DataFrame):
            if isinstance(self.features_to_transform_, pd.Index):
                return boxcox1p(X[self.features_to_transform_], self.lmbda) if not self.features_to_transform_.empty else X
            else:
                msg = "Cannot transform features {} on dataframe {}"
                raise TypeError(msg.format(get_arr_desc(self.features_to_transform_), get_arr_desc(X)))
        elif isinstance(X, pd.Series):
            return boxcox1p(X, self.lmbda) if self.features_to_transform_ else X
        elif isinstance(X, np.ndarray):
            return boxcox1p(X[:, self.features_to_transform_], self.lmbda) if self.features_to_transform_.any() else X
        # base case: if not a matched type, return if features_to_transform is "truthy"
        elif not self.features_to_transform_:
            return X
        else:
            msg = "Couldn't use boxcox transform on features in {}."
            raise TypeError(msg.format(get_arr_desc(X)))


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
