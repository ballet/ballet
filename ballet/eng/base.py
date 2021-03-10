from typing import Callable, Optional

import funcy as fy
import numpy as np
import pandas as pd
import sklearn.base
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import check_is_fitted

from ballet.exc import BalletError
from ballet.util import get_arr_desc
from ballet.util.typing import OneOrMore, TransformerLike

__all__ = (
    'BaseTransformer',
    'ConditionalTransformer',
    'GroupedFunctionTransformer',
    'GroupwiseTransformer',
    'NoFitMixin',
    'SimpleFunctionTransformer',
)


class NoFitMixin:
    """Mix-in class for transformations that do not require a fit stage"""

    def fit(self, X, y=None, **fit_kwargs):
        return self


class BaseTransformer(NoFitMixin, TransformerMixin, BaseEstimator):
    """Base transformer class for developing new transformers"""
    pass


class SimpleFunctionTransformer(FunctionTransformer):
    """Transformer that applies a callable to its input

    The callable will be called on the input X in the transform stage,
    optionally with additional arguments and keyword arguments.

    A simple wrapper around :py:class:`FunctionTransformer`.

    Args:
        func: callable to apply
        func_kwargs: keyword arguments to pass
    """

    def __init__(self,
                 func: Callable,
                 func_kwargs: Optional[dict] = None):
        self.func = func
        self.func_kwargs = func_kwargs or {}
        super().__init__(
            func=self.func,
            kw_args=self.func_kwargs)


class GroupedFunctionTransformer(FunctionTransformer):
    """Transformer that applies a callable to each group of a groupby

    Args:
        func: callable to apply
        func_kwargs: keyword arguments to pass
        groupby_kwargs: keyword arguments to ``pd.DataFrame.groupby``. If
            omitted, no grouping is performed and the function is called on
            the entire DataFrame.
    """

    def __init__(self,
                 func: Callable,
                 func_kwargs: Optional[dict] = None,
                 groupby_kwargs: Optional[dict] = None):
        self.func = func
        self.func_kwargs = func_kwargs or {}
        self.groupby_kwargs = groupby_kwargs or {}
        super().__init__(
            func=func,
            kw_args=self.func_kwargs)

    def transform(self, X, **transform_kwargs):
        if self.groupby_kwargs:
            call = X.groupby(**self.groupby_kwargs).apply
        else:
            call = X.pipe
        return call(super().transform)


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
        transformer: the transformer to apply
            to each group. If transformer is a transformer-like instance (i.e.
            has fit, transform methods etc.), then it is cloned for each group.
            If transformer is a transformer-like class (i.e. instances of
            the class are transformer-like), then it is initialized with no
            arguments for each group. If it is a callable, then it is called
            with no arguments for each group.
        groupby_kwargs: keyword arguments to pd.DataFrame.groupby
        column_selection): column, or list of columns,
            to select after the groupby. Equivalent to
            ``df.groupby(...)[column_selection]``. Defaults to None, i.e. no
            column selection is performed.
        handle_unknown: 'error' or 'ignore', default='error'. Whether to
            raise an error or ignore if an unknown group is encountered during
            transform. When this parameter is set to 'ignore' and an unknown
            group is encountered during transform, the group's values will be
            passed through unchanged.
        handle_error: 'error' or 'ignore', default='error'. Whether to
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
                 transformer: TransformerLike,
                 groupby_kwargs: dict = None,
                 column_selection: OneOrMore[str] = None,
                 handle_unknown: str = 'error',
                 handle_error: str = 'error'):
        self.satisfy_transformer = transformer
        self.groupby_kwargs = groupby_kwargs
        self.column_selection = column_selection
        self.handle_unknown = handle_unknown
        self.handle_error = handle_error

    def _make_transformer(self):
        if (
            isinstance(self.satisfy_transformer, type)
            or callable(self.satisfy_transformer)
        ):
            return self.satisfy_transformer()
        else:
            return sklearn.base.clone(self.satisfy_transformer)

    def fit(self, X, y=None, **fit_kwargs):
        # validation on inputs
        self.groupby_kwargs_ = self.groupby_kwargs or {}
        if self.handle_unknown not in ['error', 'ignore']:
            raise ValueError(
                f'Invalid value for handle_unknown: {self.handle_unknown}')
        if self.handle_error not in ['error', 'ignore']:
            raise ValueError(
                f'Invalid value for handle_error: {self.handle_error}')

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
                y_group = y[grouper.indices[group_name]]
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
                    raise BalletError(f'Unknown group: {group_name}')
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
    the satisfied columns. If a second transformation is given, applies the
    second transformation to the complement of the satisfied columns (i.e.
    the columns that fail to satisfy the condition). Otherwise, these
    unsatisfied columns are passed through unchanged.

    Args:
        condition: condition function
        satisfy_transform: transform function for satisfied columns
        unsatisfy_transform: transform function for unsatisfied columns
            (defaults to identity)
    """

    def __init__(
        self,
        condition: Callable,
        satisfy_transform: Callable,
        unsatisfy_transform: Optional[Callable] = None
    ):
        super().__init__()
        self.condition = condition
        self.satisfy_transform = satisfy_transform
        self.unsatisfy_transform = unsatisfy_transform or fy.identity

    def fit(self, X, y=None, **fit_args):
        # satisfied_columns_ is a bool or array[bool]
        self.satisfied_columns_ = self.condition(X)
        self.unsatisfied_columns_ = np.logical_not(self.satisfied_columns_)
        return self

    def transform(self, X, **transform_args):
        check_is_fitted(self, ['satisfied_columns_', 'unsatisfied_columns_'])

        if isinstance(X, pd.DataFrame):
            X = X.copy()
            X.loc[:, self.satisfied_columns_] = self.satisfy_transform(
                X.loc[:, self.satisfied_columns_])
            X.loc[:, self.unsatisfied_columns_] = self.unsatisfy_transform(
                X.loc[:, self.unsatisfied_columns_]
            )
            return X
        elif np.ndim(X) == 1:
            return (
                self.satisfy_transform(X)
                if self.satisfied_columns_
                else self.unsatisfy_transform(X)
            )
        elif isinstance(X, np.ndarray):
            X = X.copy().astype('float')
            if self.satisfied_columns_.any():
                mask = np.tile(self.satisfied_columns_, (X.shape[0], 1))
                np.putmask(X, mask, self.satisfy_transform(
                    X[:, self.satisfied_columns_]))
            if self.unsatisfied_columns_.any():
                mask = np.tile(self.unsatisfied_columns_, (X.shape[0], 1))
                np.putmask(X, mask, self.unsatisfy_transform(
                    X[:, self.unsatisfied_columns_]))
            return X
        elif not self.satisfied_columns_:
            # if we wouldn't otherwise have known what to do, we can pass
            # through X if transformation was not necessary anyways
            return self.unsatisfy_transform(X)
        else:
            raise TypeError(
                f'Couldn\'t apply transformer on features in '
                f'{get_arr_desc(X)}.')
