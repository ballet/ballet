import funcy
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = [
    'BaseTransformer',
    'GroupedFunctionTransformer',
    'NoFitMixin',
    'SimpleFunctionTransformer',
]


class NoFitMixin:
    def fit(self, X, y=None, **fit_kwargs):
        return self


class BaseTransformer(NoFitMixin, TransformerMixin, BaseEstimator):
    pass


class SimpleFunctionTransformer(BaseTransformer):
    def __init__(self, func, func_args=None, func_kwargs=None):
        super().__init__()
        self.func = func
        self.func_args = func_args if func_args else ()
        self.func_kwargs = func_kwargs if func_kwargs else {}
        self.func_call = funcy.rpartial(
            self.func, *self.func_args, **self.func_kwargs)

    def transform(self, X, **transform_kwargs):
        return self.func_call(X)


class GroupedFunctionTransformer(SimpleFunctionTransformer):
    def __init__(self, func, func_args=None,
                 func_kwargs=None, groupby_kwargs=None):
        super().__init__(func, func_args=func_args, func_kwargs=func_kwargs)
        self.groupby_kwargs = groupby_kwargs if groupby_kwargs else {}

    def transform(self, X, **transform_kwargs):
        if self.groupby_kwargs:
            call = X.sort_index().groupby(**self.groupby_kwargs).apply
        else:
            call = X.sort_index().pipe
        return call(self.func_call)


class DelegatingTransformerMixin(TransformerMixin):
    # TODO remove? almost certainly this can be accomplished with inheritance
    # directly
    def fit(self, X, y=None, **fit_args):
        return self._transformer.fit(X, y=None, **fit_args)

    def transform(self, X, **transform_args):
        return self._transformer.transform(X, **transform_args)
