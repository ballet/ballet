import unittest

import numpy as np
import pandas as pd
import sklearn.base

import ballet.eng.base
import ballet.eng.misc


class TestBase(unittest.TestCase):
    def setUp(self):
        pass

    def test_no_fit_mixin(self):
        class _Foo(ballet.eng.base.NoFitMixin):
            pass
        a = _Foo()

        self.assertTrue(hasattr(a, 'fit'))

        # method should exist
        a.fit('X')
        a.fit('X', y=None)


    def test_base_transformer(self):
        a = ballet.eng.base.BaseTransformer()

        self.assertIsInstance(a, sklearn.base.BaseEstimator)
        self.assertTrue(hasattr(a, 'fit'))

    def test_simple_function_transformer(self):
        def func(x): return x + 5
        data = np.arange(30)

        trans = ballet.eng.base.SimpleFunctionTransformer(func)
        trans.fit(data)
        data_trans = trans.transform(data)
        data_func = func(data)

        self.assertTrue(np.array_equal(data_trans, data_func))

    def test_grouped_function_transformer(self):
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
        trans = ballet.eng.base.GroupedFunctionTransformer(
            func, groupby_kwargs={'level': 'country'})
        trans.fit(df)
        result = trans.transform(df)
        expected_result = df.groupby(level='country').apply(func)
        pd.util.testing.assert_frame_equal(result, expected_result)

        # without groupby kwargs, produces a series
        func = np.min
        trans = ballet.eng.base.GroupedFunctionTransformer(func)
        trans.fit(df)
        result = trans.transform(df)
        expected_result = df.pipe(func)
        pd.util.testing.assert_series_equal(result, expected_result)


class TestMisc(unittest.TestCase):
    def setup(self):
        pass

    def test_value_replacer(self):
        trans = ballet.eng.misc.ValueReplacer(0.0, -99)
        data = pd.DataFrame([0, 0, 0, 0, 1, 3, 7, 11, -7])
        expected_result = pd.DataFrame([-99, -99, -99, -99, 1, 3, 7, 11, -7])

        result = trans.fit_transform(data)
        pd.util.testing.assert_frame_equal(result, expected_result)

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
