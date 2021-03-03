import numpy as np
import pandas as pd
import pytest
from scipy.special import boxcox1p

import ballet.eng.misc
from ballet.util.testing import assert_array_almost_equal, assert_array_equal


def test_value_replacer():
    trans = ballet.eng.misc.ValueReplacer(0.0, -99)
    data = pd.DataFrame([0, 0, 0, 0, 1, 3, 7, 11, -7])
    expected_result = pd.DataFrame([-99, -99, -99, -99, 1, 3, 7, 11, -7])

    result = trans.fit_transform(data)
    pd.util.testing.assert_frame_equal(result, expected_result)


def test_box_cox_transformer():
    threshold = 0.0
    lmbda = 0.0
    trans = ballet.eng.misc.BoxCoxTransformer(
        threshold=threshold, lmbda=lmbda)

    skewed = [0., 0., 0., 0., 1.]
    unskewed = [0., 0., 0., 0., 0.]

    exp_skew_res = boxcox1p(skewed, lmbda)
    exp_unskew_res = unskewed

    # test on DF, one skewed column
    df = pd.DataFrame()
    df['skewed'] = skewed
    df['unskewed'] = unskewed
    df_res = trans.fit_transform(df)
    assert isinstance(df_res, pd.DataFrame)
    assert 'skewed' in df_res.columns
    assert 'unskewed' in df_res.columns
    assert_array_almost_equal(df_res['skewed'], exp_skew_res)

    # test on DF, no skewed columns
    df_unskewed = pd.DataFrame()
    df_unskewed['col1'] = unskewed
    df_unskewed['col2'] = unskewed
    df_unskewed_res = trans.fit_transform(df_unskewed)
    assert isinstance(df_unskewed_res, pd.DataFrame)
    assert 'col1' in df_unskewed_res.columns
    assert 'col2' in df_unskewed_res.columns
    assert_array_equal(df_unskewed_res['col1'], exp_unskew_res)
    assert_array_equal(df_unskewed_res['col2'], exp_unskew_res)

    # test on skewed Series
    ser_skewed = pd.Series(skewed)
    ser_skewed_res = trans.fit_transform(ser_skewed)
    assert isinstance(ser_skewed_res, pd.Series)
    assert_array_almost_equal(ser_skewed_res, exp_skew_res)

    # test on unskewed Series
    ser_unskewed = pd.Series(unskewed)
    ser_unskewed_res = trans.fit_transform(ser_unskewed)
    assert isinstance(ser_unskewed_res, pd.Series)
    assert_array_equal(ser_unskewed_res, exp_unskew_res)

    # test on array
    arr = np.array([unskewed, skewed]).T
    arr_res = trans.fit_transform(arr)
    arr_exp = np.vstack((exp_unskew_res, exp_skew_res)).T
    assert_array_almost_equal(arr_res, arr_exp)


def test_named_framer():
    name = 'foo'

    # good objects
    index = pd.Index([1, 2, 3])
    ser = pd.Series([1, 2, 3])
    df = pd.DataFrame([1, 2, 3])
    arr = np.array([1, 2, 3])

    for obj in [index, ser, df, arr]:
        trans = ballet.eng.misc.NamedFramer(name)
        result = trans.fit_transform(obj)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
        assert result.columns == [name]

    # bad objects -- wrong type
    int_ = 1
    str_ = 'hello'
    for obj in [int_, str_]:
        trans = ballet.eng.misc.NamedFramer(name)
        with pytest.raises(TypeError):
            result = trans.fit_transform(obj)

    # bad objects -- too wide
    wide_df = pd.DataFrame(np.arange(10).reshape(-1, 2))
    wide_arr = np.arange(10).reshape(-1, 2)
    for obj in [wide_df, wide_arr]:
        trans = ballet.eng.misc.NamedFramer(name)
        with pytest.raises(ValueError):
            result = trans.fit_transform(obj)
