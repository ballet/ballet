import numpy as np
import pandas as pd
import pytest
import sklearn.base
from sklearn.impute import SimpleImputer

import ballet.eng
import ballet.exc
from ballet.util.testing import (
    assert_array_equal, assert_frame_equal, assert_series_equal,
    assert_series_not_equal,)


def test_no_fit_mixin():
    class _Foo(ballet.eng.NoFitMixin):
        pass
    a = _Foo()

    assert hasattr(a, 'fit')

    # method should exist
    a.fit('X')
    a.fit('X', y=None)


def test_base_transformer():
    a = ballet.eng.BaseTransformer()

    assert isinstance(a, sklearn.base.BaseEstimator)
    assert hasattr(a, 'fit')


def test_simple_function_transformer():
    def func(x): return x + 5
    data = np.arange(30)

    trans = ballet.eng.SimpleFunctionTransformer(func)
    trans.fit(data)
    data_trans = trans.transform(data)
    data_func = func(data)

    assert_array_equal(data_trans, data_func)


def test_simple_function_transformer_str_repr():
    trans = ballet.eng.SimpleFunctionTransformer(lambda x: x)
    for func in [str, repr]:
        s = func(trans)
        assert len(s) > 0


def test_grouped_function_transformer():
    df = pd.DataFrame(
        data={
            'country': ['USA', 'USA', 'USA', 'Canada', 'Fiji'],
            'year': [2001, 2002, 2003, 2001, 2001],
            'length': [1, 2, 3, 4, 5],
            'width': [1.0, 1.0, 7.5, 9.0, 11.0],
        }
    ).set_index(['country', 'year']).sort_index()

    # with groupby kwargs, produces a df
    func = np.sum
    trans = ballet.eng.GroupedFunctionTransformer(
        func, groupby_kwargs={'level': 'country'})
    trans.fit(df)
    result = trans.transform(df)
    expected_result = df.groupby(level='country').apply(func)
    assert_frame_equal(result, expected_result)

    # without groupby kwargs, produces a series
    func = np.min
    trans = ballet.eng.GroupedFunctionTransformer(func)
    trans.fit(df)
    result = trans.transform(df)
    expected_result = df.pipe(func)
    assert_series_equal(result, expected_result)


@pytest.fixture
def sample_data():
    X_tr = pd.DataFrame(
        data={
            'name': ['A', 'A', 'A', 'B', 'B', 'C', 'C'],
            'year': [2001, 2002, 2003, 2001, 2002, 2001, 2003],
            'value': [1, 2, np.nan, 4, 4, 5, np.nan],
            'size': [3, 5, 5, 5, 5, np.nan, 4],
        }
    ).set_index(['name', 'year']).sort_index()

    X_te = pd.DataFrame(
        data={
            'name': ['A', 'B', 'C'],
            'year': [2004, 2004, 2004],
            'value': [np.nan, 1.5, np.nan],
            'size': [4, 1, np.nan],
        }
    ).set_index(['name', 'year']).sort_index()

    return X_tr, X_te


@pytest.fixture
def groupby_kwargs():
    return {'level': 'name'}


@pytest.fixture
def individual_transformer():
    return SimpleImputer()


@pytest.fixture
def groupwise_transformer(groupby_kwargs, individual_transformer):
    # mean-impute within groups
    return ballet.eng.GroupwiseTransformer(
        individual_transformer,
        groupby_kwargs=groupby_kwargs,
        column_selection=['value'],
    )


def test_groupwise_transformer_can_fit(sample_data, groupwise_transformer):
    X_tr, X_te = sample_data
    groupwise_transformer.fit(X_tr)


def test_groupwise_transformer_can_transform(
    sample_data, groupwise_transformer
):
    X_tr, X_te = sample_data
    groupwise_transformer.fit(X_tr)

    result_tr = groupwise_transformer.transform(X_tr)
    expected_tr = X_tr.copy()
    expected_tr['value'] = np.array([1, 2, 1.5, 4, 4, 5, 5])
    expected_tr = expected_tr.drop('size', axis=1)
    assert_frame_equal(result_tr, expected_tr)

    result_te = groupwise_transformer.transform(X_te)
    expected_te = X_te.copy()
    expected_te['value'] = np.array([1.5, 1.5, 5])
    expected_te = expected_te.drop('size', axis=1)
    assert_frame_equal(result_te, expected_te)


def test_groupwise_transformer_raise_on_new_group(
    sample_data, individual_transformer, groupby_kwargs
):
    X_tr, X_te = sample_data

    # mean-impute within groups
    groupwise_transformer = ballet.eng.GroupwiseTransformer(
        individual_transformer,
        groupby_kwargs=groupby_kwargs,
        handle_unknown='error',
    )

    groupwise_transformer.fit(X_tr)

    X_te = X_te.copy().reset_index()
    X_te.loc[0, 'name'] = 'Z'  # new group
    X_te = X_te.set_index(['name', 'year'])

    with pytest.raises(ballet.exc.BalletError):
        groupwise_transformer.transform(X_te)


def test_groupwise_transformer_ignore_on_new_group(
    sample_data, individual_transformer, groupby_kwargs
):

    X_tr, X_te = sample_data

    groupwise_transformer = ballet.eng.GroupwiseTransformer(
        individual_transformer,
        groupby_kwargs=groupby_kwargs,
        handle_unknown='ignore',
    )

    groupwise_transformer.fit(X_tr)

    X_te = X_te.copy().reset_index()
    X_te.loc[0, 'name'] = 'Z'  # new group
    X_te = X_te.set_index(['name', 'year'])

    result = groupwise_transformer.transform(X_te)

    # the first group, Z, is new, and values are passed through, so such
    # be nan
    expected = X_te.copy()
    expected['value'] = np.array([np.nan, 1.5, 5.0])
    expected['size'] = np.array([4.0, 1.0, 4.0])

    assert_frame_equal(result, expected)


def test_groupwise_transformer_raise_on_transform_error(
    sample_data, groupby_kwargs
):
    X_tr, X_te = sample_data

    exc = Exception

    class TransformErrorTransformer(ballet.eng.BaseTransformer):
        def transform(self, X, **transform_kwargs):
            raise exc

    groupwise_transformer = ballet.eng.GroupwiseTransformer(
        TransformErrorTransformer(),
        groupby_kwargs=groupby_kwargs,
        handle_error='error',
    )

    groupwise_transformer.fit(X_tr)

    with pytest.raises(exc):
        groupwise_transformer.transform(X_tr)


def test_groupwise_transformer_ignore_on_transform_error(sample_data):
    X_tr, X_te = sample_data

    exc = Exception

    class TransformErrorTransformer(ballet.eng.BaseTransformer):
        def transform(self, X, **transform_kwargs):
            raise exc

    groupwise_transformer = ballet.eng.GroupwiseTransformer(
        TransformErrorTransformer(),
        groupby_kwargs={'level': 'name'},
        handle_error='ignore',
    )

    groupwise_transformer.fit(X_tr)

    result_tr = groupwise_transformer.transform(X_tr)
    expected_tr = X_tr

    assert_frame_equal(result_tr, expected_tr)

    result_te = groupwise_transformer.transform(X_te)
    expected_te = X_te
    assert_frame_equal(result_te, expected_te)


def test_conditional_transformer_both_satisfied(sample_data):
    X_tr, X_te = sample_data

    t = ballet.eng.ConditionalTransformer(
        lambda ser: ser.sum() > 0,
        lambda ser: ser + 1,
    )

    # all the features are selected by sum > 0
    t.fit(X_tr)
    result_tr = t.transform(X_tr)
    for col in ['value', 'size']:
        assert_series_not_equal(result_tr[col], X_tr[col])

    result_te = t.transform(X_te)
    for col in ['value', 'size']:
        assert_series_not_equal(result_te[col], X_te[col])


def test_conditional_transformer_one_satisfied(sample_data):
    X_tr, X_te = sample_data

    t = ballet.eng.ConditionalTransformer(
        lambda ser: (ser.dropna() >= 3).all(),
        lambda ser: ser.fillna(0) + 1,
    )

    t.fit(X_tr)
    result_tr = t.transform(X_tr)
    result_te = t.transform(X_te)

    # only 'size' is selected by the condition
    assert_series_not_equal(result_tr['size'], X_tr['size'])
    assert_series_not_equal(result_te['size'], X_te['size'])

    # 'value' is not selected by the condition, has items less than 3
    assert_series_equal(result_tr['value'], X_tr['value'])
    assert_series_equal(result_te['value'], X_te['value'])


def test_conditional_transformer_unsatisfy_transform(sample_data):
    X_tr, X_te = sample_data

    t = ballet.eng.ConditionalTransformer(
        lambda ser: (ser.dropna() >= 3).all(),
        lambda ser: ser,
        lambda ser: ser.fillna(0) - 1,
    )

    t.fit(X_tr)
    result_tr = t.transform(X_tr)
    result_te = t.transform(X_te)

    # size is transformed by satisfy condition, but passed through
    assert_series_equal(result_tr['size'], X_tr['size'])
    assert_series_equal(result_te['size'], X_te['size'])

    # value is transformed by unsatisfy condition and is not equal
    assert_series_not_equal(result_tr['value'], X_tr['value'])
    assert_series_not_equal(result_te['value'], X_te['value'])
