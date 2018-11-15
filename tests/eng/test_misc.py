import math
import unittest

import numpy as np
import pandas as pd

import ballet.eng.misc
from ballet.util.testing import ArrayLikeEqualityTestingMixin

class TestMisc(ArrayLikeEqualityTestingMixin, unittest.TestCase):
    def setup(self):
        pass

    def test_value_replacer(self):
        trans = ballet.eng.misc.ValueReplacer(0.0, -99)
        data = pd.DataFrame([0, 0, 0, 0, 1, 3, 7, 11, -7])
        expected_result = pd.DataFrame([-99, -99, -99, -99, 1, 3, 7, 11, -7])

        result = trans.fit_transform(data)
        pd.util.testing.assert_frame_equal(result, expected_result)

    def test_box_cox_transformer(self):
        a = ballet.eng.misc.BoxCoxTransformer(threshold=0)

        exp_skew_res = np.log1p([0,0,0,0,1])
        exp_unskew_res = [0,0,0,0,0]

        # test on DF, one skewed column
        df = pd.DataFrame()
        df['skewed'] = [0,0,0,0,1]
        df['unskewed'] = [0,0,0,0,0]
        df_res = a.fit_transform(df)
        self.assertTrue(isinstance(df_res, pd.DataFrame))
        self.assertTrue('skewed' in df_res.columns)
        self.assertTrue('unskewed' not in df_res.columns)
        self.assertArrayAlmostEqual(df_res['skewed'], exp_skew_res)

        # test on DF, no skewed columns
        df_unskewed = pd.DataFrame()
        df_unskewed['col1'] = [0,0,0,0,0]
        df_unskewed['col2'] = [0,0,0,0,0]
        df_unskewed_res = a.fit_transform(df_unskewed)
        self.assertTrue(isinstance(df_unskewed_res, pd.DataFrame))
        self.assertTrue('col1' in df_unskewed_res.columns)
        self.assertTrue('col2' in df_unskewed_res.columns)
        self.assertArrayEqual(df_unskewed_res['col1'], exp_unskew_res)
        self.assertArrayEqual(df_unskewed_res['col2'], exp_unskew_res)


        # test on skewed Series
        ser_skewed = pd.Series([0,0,0,0,1])
        ser_skewed_res = a.fit_transform(ser_skewed)
        self.assertTrue(isinstance(ser_skewed_res, pd.Series))
        self.assertArrayAlmostEqual(ser_skewed_res, exp_skew_res)

        # test on unskewed Series
        ser_unskewed = pd.Series([0,0,0,0,0])
        ser_unskewed_res = a.fit_transform(ser_unskewed)
        self.assertTrue(isinstance(ser_unskewed_res, pd.Series))
        self.assertArrayEqual(ser_unskewed_res, exp_unskew_res)

        # test on np arrs
        nparr = np.array([[0,0],[0,0],[0,0],[0,0],[0,1]])
        np_res = a.fit_transform(nparr)
        self.assertArrayAlmostEqual(np.reshape(np_res, 5), exp_skew_res)

    def test_named_framer(self):
        name = 'foo'

        # good objects
        index = pd.Index([1, 2, 3])
        ser = pd.Series([1, 2, 3])
        df = pd.DataFrame([1, 2, 3])
        arr = np.array([1, 2, 3])

        for obj in [index, ser, df, arr]:
            trans = ballet.eng.misc.NamedFramer(name)
            result = trans.fit_transform(obj)
            self.assertTrue(isinstance(result, pd.DataFrame))
            self.assertEqual(result.shape[1], 1)
            self.assertEqual(result.columns, [name])

        # bad objects -- wrong type
        int_ = 1
        str_ = 'hello'
        for obj in [int_, str_]:
            trans = ballet.eng.misc.NamedFramer(name)
            with self.assertRaises(TypeError):
                result = trans.fit_transform(obj)

        # bad objects -- too wide
        wide_df = pd.DataFrame(np.arange(10).reshape(-1, 2))
        wide_arr = np.arange(10).reshape(-1, 2)
        for obj in [wide_df, wide_arr]:
            trans = ballet.eng.misc.NamedFramer(name)
            with self.assertRaises(ValueError):
                result = trans.fit_transform(obj)
