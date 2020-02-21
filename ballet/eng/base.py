import funcy
import numpy as np
import pandas as pd
import sklearn.base
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ballet.exc import BalletError
from ballet.util import get_arr_desc

__all__ = [
    'BaseTransformer',
    'ConditionalTransformer',
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
        args = funcy.join((args, func_args))  # py34
        return self.func(*args, **func_kwargs)

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

    For each group identified in the training set by the groupby operation,
    a separate transformer is cloned and fit. This is useful to learn
    group-wise transformers that do not leak data between the training and
    test sets. Consider the case of imputing missing values with the mean of
    some group. A normal, pure-pandas implementation, such as
    ``X_te.groupby(by='foo').apply('mean')`` would leak information about
    the test set means, which might differ from the training set means.

    Args:
        transformer (transformer-like or callable): the transformer to apply
            to each group. If transformer is a transformer-like instance (i.e.
            has fit, transform methods etc.), then it is cloned for each group.
            If transformer is a transformer-like class (i.e. instances of
            the class are transformer-like), then it is initialized with no
            arguments for each group. If it is a callable, then it is called
            with no arguments for each group.
        groupby_kwargs (dict): keyword arguments to pd.DataFrame.groupby
        column_selection (str, List[str]): column, or list of columns,
            to select after the groupby. Equivalent to
            `df.groupby(...)[column_selection]`. Defaults to None, i.e. no
            column selection is performed.
        handle_unknown (str): 'error' or 'ignore', default=’error’. Whether to
            raise an error or ignore if an unknown group is encountered during
            transform. When this parameter is set to 'ignore' and an unknown
            group is encountered during transform, the group's values will be
            passed through unchanged.
        handle_error (str): 'error' or 'ignore', default='error'. Whether to
            raise an error or ignore if an error is raised during transforming
            an individual group. When this parameter is set to 'ignore' and
            an error is raised when calling the transformer's transform
            method on an individual group, the group's values will be passed
            through unchanged.

    Example usage:

        In this example, we create a groupwise transformer that fits a
        separate imputer for each group encountered. For new data points,
        values will be imputed according to the mean of its group on the
        training set, avoiding any data leakage.

        .. code-block:: python

           >>> from sklearn.impute import SimpleImputer
           >>> transformer = GroupwiseTransformer(
           ...     SimpleImputer(strategy='mean'),
           ...     groupby_kwargs = {'level': 'name'}
           ... )

    Raises:
        ballet.exc.BalletError: if handle_unknown=='error' and an unknown group
            is encountered at transform-time.
    """

    def __init__(self,
                 transformer,
                 groupby_kwargs=None,
                 column_selection=None,
                 handle_unknown='error',
                 handle_error='error'):
        self.transformer = transformer
        self.groupby_kwargs = groupby_kwargs
        self.column_selection = column_selection
        self.handle_unknown = handle_unknown
        self.handle_error = handle_error

    def _make_transformer(self):
        if isinstance(self.transformer, type) or callable(self.transformer):
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
        if self.handle_error not in ['error', 'ignore']:
            raise ValueError(
                'Invalid value for handle_error: {}'
                .format(self.handle_error))

        # Get the groups
        grouper = X.groupby(**self.groupby_kwargs_)
        self.groups_ = set(grouper.groups.keys())

        # Create and fit a transformer for each group
        self.transformers_ = {}
        for group_name, x_group in grouper:
            transformer = self._make_transformer()

            if self.column_selection is not None:
                x_group = x_group[self.column_selection]

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
            # If the group is not a DataFrame, there are two problems
            # 1. We can't rely on group.name to lookup the right transformer
            # 2. We can't "reassemble" the transformed
            # However, the contract of ``pandas.core.groupby.GroupBy.apply`` is
            # that the input is a DataFrame, so this should never occur.
            if not isinstance(x_group, pd.DataFrame):
                raise NotImplementedError

            group_name = x_group.name

            if self.column_selection is not None:
                x_group = x_group[self.column_selection]

            if group_name in self.transformers_:
                transformer = self.transformers_[group_name]
                try:
                    data = transformer.transform(x_group, *args, **kwargs)

                    # This post-processing step is required because sklearn
                    # transform converts a DataFrame to an array. This is my
                    # best attempt so far to approximate the following:
                    # >>> result = x_group.copy()
                    # >>> result.values = data
                    # which is an error as `values` cannot be set.
                    index = x_group.index
                    columns = x_group.columns
                    return pd.DataFrame(
                        data=data, index=index, columns=columns)
                except Exception:
                    if self.handle_error == 'ignore':
                        return x_group
                    else:
                        raise
            else:
                if self.handle_unknown == 'error':
                    raise BalletError(
                        'Unknown group: {group_name}'
                        .format(group_name=group_name))
                elif self.handle_unknown == 'ignore':
                    return x_group
                else:
                    # Unreachable code
                    raise RuntimeError

        return (
            X
            .groupby(**self.groupby_kwargs_)
            .apply(_transform, **transform_kwargs)
        )


class ConditionalTransformer(BaseTransformer):
    """Transform columns that satisfy a condition during training

    In the fit stage, determines which variables (columns) satisfy the
    condition. In the transform stage, applies the given transformation to
    the selected columns, passing through the complement unchanged.

    Args:
        condition (callable): condition function
        trans (callable): transform function
    """

    def __init__(self, condition, trans):
        super().__init__()
        self.condition = condition
        # 1. can't call it transform because conflicts with method name
        # 2. can't call it transform and assign to a different attribute
        # because sklearn.clone would complain
        self.trans = trans

    def fit(self, X, y=None, **fit_args):
        # features_to_transform_ is a bool or array[bool]
        self.features_to_transform_ = self.condition(X)
        return self

    def transform(self, X, **transform_args):
        check_is_fitted(self, 'features_to_transform_')

        if isinstance(X, pd.DataFrame):
            X = X.copy()
            X.loc[:, self.features_to_transform_] = self.trans(
                X.loc[:, self.features_to_transform_])
            return X
        elif np.ndim(X) == 1:
            return self.trans(X) if self.features_to_transform_ else X
        elif isinstance(X, np.ndarray):
            if self.features_to_transform_.any():
                X = X.copy()
                X = X.astype('float')
                mask = np.tile(self.features_to_transform_, (X.shape[0], 1))
                np.putmask(X, mask, self.trans(
                    X[:, self.features_to_transform_]))
            return X
        elif not self.features_to_transform_:
            # if we wouldn't otherwise have known what to do, we can pass
            # through X if transformation was not necessary anyways
            return X
        else:
            raise TypeError(
                "Couldn't apply transformer on features in {}."
                .format(get_arr_desc(X)))
