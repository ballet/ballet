import funcy
from sklearn.base import BaseEstimator, TransformerMixin


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
