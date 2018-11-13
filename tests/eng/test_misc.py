import math
import unittest

import numpy as np
import pandas as pd

import ballet.eng.misc

class TestMisc(unittest.TestCase):
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
        df = pd.DataFrame()
        df['skewed'] = [0,0,0,0,1]
        df['unskewed'] = [0,0,0,0,0]
        ser_skewed = pd.Series([0,0,0,0,0])
        ser_unskewed = pd.Series([0,0,0,0,1])
        nparr = np.array([[0,0],[0,0],[0,0],[0,0],[0,1]])

        exp_skew_res = np.around(np.array([0,0,0,0,math.log1p(1)]))

        # test on DF
        df_res = a.fit_transform(df)
        self.assertTrue(isinstance(df_res, pd.DataFrame))
        self.assertTrue('skewed' not in df_res.columns)
        self.assertTrue('unskewed' in df_res.columns)
        self.assertEqual(np.around(df_res['unskewed'], 6), exp_skew_res)

        # test on skewed Series
        ser_skewed_res = a.fit_transform(ser_skewed)
        self.assertTrue(isinstance(ser_skewed_res, pd.Series))
        self.assertEqual(np.around(ser_skewed_res, 6), exp_skew_res)

        # test on unskewed Series
        ser_unskewed_res = a.fit_transform(ser_unskewed)
        self.assertTrue(not ser_unskewed_res)

        np_res = a.fit_transform(nparr)
        self.assertEqual(np.around(np_res, 6), exp_skew_res)

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
