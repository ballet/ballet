import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelBinarizer


class FeatureTypeTransformer(BaseEstimator, TransformerMixin):
    BAD_TYPE_MSG = "Unsupported input type '{}'"
    BAD_SHAPE_MSG = "Unsupported input shape '{}'"

    @staticmethod
    def _get_original_info(X):
        if isinstance(X, pd.Series):
            return {
                'index': X.index,
                'dtype': X.dtype,
                'name': X.name,
            }
        elif isinstance(X, pd.DataFrame):
            return {
                'index': X.index,
                'dtypes': X.dtypes,
                'columns': X.columns,
            }
        elif isinstance(X, np.ndarray):
            return {'ndim': X.ndim}
        else:
            return {}

    def fit(self, X, **fit_kwargs):
        self.original_type_ = type(X)
        self.original_info_ = self._get_original_info(X)
        return self

    def transform(self, X, **transform_kwargs):
        if not hasattr(self, 'original_type_'):
            raise NotFittedError

        if isinstance(X, pd.Series):
            return X.values
            # return X.to_frame().to_records(index=False))
        elif isinstance(X, pd.DataFrame):
            return X.values
            # return np.asarray(X.to_records(index=False))
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                return X.reshape(-1, 1)
            elif X.ndim == 2:
                return X
            elif X.ndim >= 3:
                raise TypeError(
                    FeatureTypeTransformer.BAD_SHAPE_MSG.format(
                        X.shape))
        else:
            # should be unreachable
            raise TypeError(
                FeatureTypeTransformer.BAD_TYPE_MSG.format(
                    type(X)))

    def inverse_transform(self, X, **inverse_transform_kwargs):
        if not hasattr(self, 'original_type_'):
            raise NotFittedError

        if hasattr(self, 'original_type_') and hasattr(self, 'original_info_'):
            if issubclass(self.original_type_, pd.Series):
                data = X
                index = self.original_info_['index']
                name = self.original_info_['name']
                dtype = self.original_info_['dtype']
                return pd.Series(data=data, index=index,
                                 name=name, dtype=dtype)
            elif issubclass(self.original_type_, pd.DataFrame):
                data = X
                index = self.original_info_['index']
                columns = self.original_info_['columns']
                dtypes = self.original_info_['dtypes']
                df = pd.DataFrame(data=data, index=index,
                                  columns=columns)
                df = df.astype(dtype=dtypes.to_dict())
                return df
            elif issubclass(self.original_type_, np.ndarray):
                # only thing we might have done is change dimensions for 1d/2d
                if self.original_info_['ndim'] == 1:
                    return X.ravel()
                elif self.original_info_['ndim'] == 2:
                    return X
            else:
                # unreachable
                raise NotImplementedError
        else:
            raise NotFittedError


class TargetTypeTransformer(FeatureTypeTransformer):
    def __init__(self, needs_label_binarizer=False):
        super().__init__()
        self.needs_label_binarizer = needs_label_binarizer

    def fit(self, y, **fit_kwargs):
        super().fit(y, **fit_kwargs)
        if self.needs_label_binarizer:
            self.label_binarizer_ = LabelBinarizer()
            self.label_binarizer_.fit(y)
        return self

    def transform(self, y, **transform_kwargs):
        y = super().transform(y)
        if self.needs_label_binarizer:
            y = self.label_binarizer_.transform(y)
        else:
            y = y.ravel()
        return y

    def inverse_transform(self, y, **inverse_transform_kwargs):
        if self.needs_label_binarizer:
            y = self.label_binarizer_.inverse_transform(y)
        y = super().inverse_transform(y)
        return y


