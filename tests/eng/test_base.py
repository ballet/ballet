import unittest

import numpy as np
import pandas as pd
import sklearn.base

import ballet.eng.base


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