import pandas as pd
import sklearn.base
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ballet.exc import Error

__all__ = [
    'BaseTransformer',
    'GroupedFunctionTransformer',
    'GroupwiseTransformer',
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
        self.func_args = func_args
        self.func_kwargs = func_kwargs

    def _func_call(self, *args):
        func_args = self.func_args or ()
        func_kwargs = self.func_kwargs or {}
        return self.func(*args, *func_args, **func_kwargs)

    def transform(self, X, **transform_kwargs):
        return self._func_call(X)


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
        self.groupby_kwargs = groupby_kwargs

    def transform(self, X, **transform_kwargs):
        groupby_kwargs = self.groupby_kwargs or {}
        if groupby_kwargs:
            call = X.groupby(**groupby_kwargs).apply
        else:
            call = X.pipe
        return call(self._func_call)


class GroupwiseTransformer(BaseTransformer):
    """Transformer that does something different for every group

    Args:

        transformer (transformer-like or callable): the transformer to apply
            to each group. If transformer is a transformer-like instance (i.e.
            has fit, transform methods etc.), then it is cloned for each group.
            If transformer is a transformer-like class (i.e. instances of
            the class are transformer-like), then it is initialized with no
            arguments for each group. If it is a callable, then it is called
            with no arguments for each group.

        handle_unknown: 'error' or 'ignore', default=’error’. Whether to raise
            an error or ignore if an unknown categorical feature is present
            during transform (default is to raise). When this parameter is set
            to 'ignore' and an unknown category is encountered during
            transform, the resulting one-hot encoded columns for this feature
            will be all zeros. In the inverse transform, an unknown category
            will be denoted as None.
    """

    def __init__(self,
                 transformer,
                 groupby_kwargs=None,
                 handle_unknown='error'):
        self.transformer = transformer
        self.groupby_kwargs = groupby_kwargs
        self.handle_unknown = handle_unknown

    def _make_transformer(self):
        if type(self.transformer) is type or callable(self.transformer):
            return self.transformer()
        else:
            return sklearn.base.clone(self.transformer)

    def fit(self, X, y=None, **fit_kwargs):
        # validation on inputs
        self.groupby_kwargs_ = self.groupby_kwargs or {}
        if self.handle_unknown not in ['error', 'ignore']:
            raise ValueError(
                'Invalid value for handle_unknown: {}'
                .format(self.handle_unknown))

        # Get the groups
        grouper = X.groupby(**self.groupby_kwargs_)
        self.groups_ = set(grouper.groups.keys())

        # Create and fit a transformer for each group
        self.transformers_ = {}
        for group_name, x_group in grouper:
            transformer = self._make_transformer()

            if y is not None:
                # Extract y by integer indexing
                inds = grouper.indices[group_name]
                y_group = y[inds]
                transformer.fit(x_group, y_group)
            else:
                transformer.fit(x_group)

            self.transformers_[group_name] = transformer

        return self

    def transform(self, X, **transform_kwargs):
        check_is_fitted(self, ['groups_', 'transformers_'])

        def _transform(x_group, *args, **kwargs):
            if x_group.name in self.transformers_:
                transformer = self.transformers_[x_group.name]
                data = transformer.transform(x_group, *args, **kwargs)

                # this post-processing step is required because sklearn
                # transform converts a DataFrame to an array. This is my
                # best attempt so far to replicate:
                # >>> result = x_group.copy()
                # >>> result.values = data
                # which is an error as values is a protected attribute.
                # Unfortunately, this approach is not robust to different
                # data types.
                if not isinstance(x_group, pd.DataFrame):
                    raise NotImplementedError
                index = x_group.index
                columns = x_group.columns
                return pd.DataFrame(data=data, index=index, columns=columns)
            else:
                if self.handle_unknown == 'error':
                    raise Error(
                        'Unknown group: {group_name}'
                        .format(group_name=x_group.name))
                elif self.handle_unknown == 'ignore':
                    return x_group
                else:
                    # Unreachable code
                    raise RuntimeError

        return X.groupby(**self.groupby_kwargs_).apply(
            _transform, **transform_kwargs)
