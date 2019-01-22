import unittest

import numpy as np
import pandas as pd
from scipy.special import boxcox1p

from ballet.eng.misc import BoxCoxTransformer, NamedFramer, ValueReplacer
from ballet.util.testing import ArrayLikeEqualityTestingMixin


class TestMisc(ArrayLikeEqualityTestingMixin, unittest.TestCase):
    def setup(self):
        pass

    def test_value_replacer(self):
        trans = ValueReplacer(0.0, -99)
        data = pd.DataFrame([0, 0, 0, 0, 1, 3, 7, 11, -7])
        expected_result = pd.DataFrame([-99, -99, -99, -99, 1, 3, 7, 11, -7])

        result = trans.fit_transform(data)
        pd.util.testing.assert_frame_equal(result, expected_result)

    def test_value_replacer_eq(self):
        trans = ValueReplacer(0.0, -99)
        eq_trans = ValueReplacer(0.0, -99)
        self.assertEqual(trans, eq_trans)

        ne_trans_val = ValueReplacer(1.0, -99)
        self.assertNotEqual(trans, ne_trans_val)

        ne_trans_rep = ValueReplacer(0.0, -100)
        self.assertNotEqual(trans, ne_trans_rep)

    def test_box_cox_transformer(self):
        threshold = 0.0
        lmbda = 0.0
        trans = BoxCoxTransformer(threshold=threshold, lmbda=lmbda)

        skewed = [0., 0., 0., 0., 1.]
        unskewed = [0., 0., 0., 0., 0.]

        exp_skew_res = boxcox1p(skewed, lmbda)
        exp_unskew_res = unskewed

        # test on DF, one skewed column
        df = pd.DataFrame()
        df['skewed'] = skewed
        df['unskewed'] = unskewed
        df_res = trans.fit_transform(df)
        self.assertIsInstance(df_res, pd.DataFrame)
        self.assertIn('skewed', df_res.columns)
        self.assertIn('unskewed', df_res.columns)
        self.assertArrayAlmostEqual(df_res['skewed'], exp_skew_res)

        # test on DF, no skewed columns
        df_unskewed = pd.DataFrame()
        df_unskewed['col1'] = unskewed
        df_unskewed['col2'] = unskewed
        df_unskewed_res = trans.fit_transform(df_unskewed)
        self.assertIsInstance(df_unskewed_res, pd.DataFrame)
        self.assertIn('col1', df_unskewed_res.columns)
        self.assertIn('col2', df_unskewed_res.columns)
        self.assertArrayEqual(df_unskewed_res['col1'], exp_unskew_res)
        self.assertArrayEqual(df_unskewed_res['col2'], exp_unskew_res)

        # test on skewed Series
        ser_skewed = pd.Series(skewed)
        ser_skewed_res = trans.fit_transform(ser_skewed)
        self.assertIsInstance(ser_skewed_res, pd.Series)
        self.assertArrayAlmostEqual(ser_skewed_res, exp_skew_res)

        # test on unskewed Series
        ser_unskewed = pd.Series(unskewed)
        ser_unskewed_res = trans.fit_transform(ser_unskewed)
        self.assertIsInstance(ser_unskewed_res, pd.Series)
        self.assertArrayEqual(ser_unskewed_res, exp_unskew_res)

        # test on array
        arr = np.array([unskewed, skewed]).T
        arr_res = trans.fit_transform(arr)
        arr_exp = np.vstack((exp_unskew_res, exp_skew_res)).T
        self.assertArrayAlmostEqual(arr_res, arr_exp)
    
    def test_box_cox_transformer_eq(self):
        threshold = 0.0
        lmbda = 0.0
        trans = BoxCoxTransformer(threshold=threshold, lmbda=lmbda)

        eq_trans = BoxCoxTransformer(threshold=threshold, lmbda=lmbda)
        self.assertEqual(trans, eq_trans)

        ne_threshold = 1.0
        ne_trans_threshold = BoxCoxTransformer(threshold=ne_threshold, lmbda=lmbda)
        self.assertNotEqual(trans, ne_trans_threshold)

        ne_lmbda = 1.0
        ne_trans_lmbda = BoxCoxTransformer(threshold=threshold, lmbda=ne_lmbda)
        self.assertNotEqual(trans, ne_trans_lmbda)

    def test_named_framer(self):
        name = 'foo'

        # good objects
        index = pd.Index([1, 2, 3])
        ser = pd.Series([1, 2, 3])
        df = pd.DataFrame([1, 2, 3])
        arr = np.array([1, 2, 3])

        for obj in [index, ser, df, arr]:
            trans = NamedFramer(name)
            result = trans.fit_transform(obj)
            self.assertTrue(isinstance(result, pd.DataFrame))
            self.assertEqual(result.shape[1], 1)
            self.assertEqual(result.columns, [name])

        # bad objects -- wrong type
        int_ = 1
        str_ = 'hello'
        for obj in [int_, str_]:
            trans = NamedFramer(name)
            with self.assertRaises(TypeError):
                result = trans.fit_transform(obj)

        # bad objects -- too wide
        wide_df = pd.DataFrame(np.arange(10).reshape(-1, 2))
        wide_arr = np.arange(10).reshape(-1, 2)
        for obj in [wide_df, wide_arr]:
            trans = NamedFramer(name)
            with self.assertRaises(ValueError):
                result = trans.fit_transform(obj)

    def test_named_framer_eq(self):
        name = 'foo'
        ne_name = 'bar'
        trans = NamedFramer(name)
        eq_trans = NamedFramer(name)
        ne_trans = NamedFramer(ne_name)
        self.assertEqual(trans, eq_trans)
        self.assertNotEqual(trans, ne_trans)

