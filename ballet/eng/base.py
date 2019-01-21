import funcy
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = [
    'BaseTransformer',
    'GroupedFunctionTransformer',
    'NoFitMixin',
    'SimpleFunctionTransformer',
]


class NoFitMixin:
    """Mix-in class for transformations that do not require a fit stage"""
    def fit(self, X, y=None, **fit_kwargs):
        return self


class BaseTransformer(NoFitMixin, TransformerMixin, BaseEstimator):
    """Base transformer class for developing new transformers"""
    pass


class SimpleFunctionTransformer(BaseTransformer):
    """Transformer that applies a callable to its input

    The callable will be called on the input X in the transform stage,
    optionally with additional arguments and keyword arguments.

    This will be eventually replaced with
    ``sklearn.preprocessing.FunctionTransformer``.

    Args:
        func (callable): callable to apply
        func_args (tuple): additional arguments to pass
        func_kwargs (dict): keyword arguments to pass
    """
    def __init__(self, func, func_args=None, func_kwargs=None):
        super().__init__()
        self.func = func
        self.func_args = func_args if func_args else ()
        self.func_kwargs = func_kwargs if func_kwargs else {}
        self.func_call = funcy.rpartial(
            self.func, *self.func_args, **self.func_kwargs)
    
    def __eq__(self, other):
        if not isinstance(other, SimpleFunctionTransformer):
            return False 
        if self.func.__code__.co_code != other.func.__code__.co_code:
            return False
        if self.func_args != other.func_args:
            return False
        if self.func_kwargs != other.func_kwargs:
            return False
        return True

    def transform(self, X, **transform_kwargs):
        return self.func_call(X)


class GroupedFunctionTransformer(SimpleFunctionTransformer):
    """Transformer that applies a callable to each group of a groupby

    Args:
        func (callable): callable to apply
        func_args (tuple): additional arguments to pass
        func_kwargs (dict): keyword arguments to pass
        groupby_kwargs (dict): keyword arguments to pd.DataFrame.groupby
    """
    def __init__(self, func, func_args=None,
                 func_kwargs=None, groupby_kwargs=None):
        super().__init__(func, func_args=func_args, func_kwargs=func_kwargs)
        self.groupby_kwargs = groupby_kwargs if groupby_kwargs else {}
    
    def __eq__(self, other):
        if not isinstance(other, GroupedFunctionTransformer):
            return False
        if not SimpleFunctionTransformer.__eq__(self, other):
            return False
        if self.groupby_kwargs != other.groupby_kwargs:
            return False
        return True

    def transform(self, X, **transform_kwargs):
        if self.groupby_kwargs:
            call = X.sort_index().groupby(**self.groupby_kwargs).apply
        else:
            call = X.sort_index().pipe
        return call(self.func_call)
