import unittest

import funcy
import numpy as np
import pandas as pd
import sklearn.preprocessing

from ballet.compat import SimpleImputer
from ballet.eng.misc import IdentityTransformer
from ballet.feature import Feature
from ballet.pipeline import FeatureEngineeringPipeline
from ballet.transformer import DelegatingRobustTransformer
from ballet.util import asarray2d

from .util import FragileTransformer, FragileTransformerPipeline


class DelegatingRobustTransformerTest(unittest.TestCase):

    def setUp(self):
        self.X_ser = pd.util.testing.makeFloatSeries()
        self.X_df = self.X_ser.to_frame()
        self.X_arr1d = np.asarray(self.X_ser)
        self.X_arr2d = np.asarray(self.X_df)
        self.y_ser = self.X_ser.copy()
        self.y_df = self.X_df.copy()
        self.y_arr1d = np.asarray(self.y_ser)
        self.y_arr2d = np.asarray(self.y_df)

        self.d = {
            'ser': (self.X_ser, self.y_ser),
            'df': (self.X_df, self.y_df),
            'arr1d': (self.X_arr1d, self.y_arr1d),
            'arr2d': (self.X_arr2d, self.y_arr2d),
        }

    def _test_robust_transformer(
        self, input_types, bad_input_checks, catches,
        transformer_maker=FragileTransformer
    ):
        fragile_transformer = transformer_maker(bad_input_checks, catches)
        robust_transformer = DelegatingRobustTransformer(
            transformer_maker(bad_input_checks, catches))

        for input_type in input_types:
            X, y = self.d[input_type]
            # fragile transformer raises error
            with self.assertRaises(catches):
                fragile_transformer.fit_transform(X, y)
            # robust transformer does not raise error
            X_robust = robust_transformer.fit_transform(X, y)
            self.assertTrue(
                np.array_equal(
                    asarray2d(X),
                    asarray2d(X_robust)
                )
            )

    def test_robust_transformer_ser(self):
        input_types = ('ser',)
        bad_input_checks = (
            lambda x: isinstance(x, pd.Series),
        )
        catches = (ValueError, TypeError)
        self._test_robust_transformer(input_types, bad_input_checks, catches)

    def test_robust_transformer_df(self):
        input_types = ('ser', 'df',)
        bad_input_checks = (
            lambda x: isinstance(x, pd.Series),
            lambda x: isinstance(x, pd.DataFrame),
        )
        catches = (ValueError, TypeError)
        self._test_robust_transformer(input_types, bad_input_checks, catches)

    def test_robust_transformer_arr(self):
        input_types = ('ser', 'df', 'arr1d')
        bad_input_checks = (
            lambda x: isinstance(x, pd.Series),
            lambda x: isinstance(x, pd.DataFrame),
            lambda x: isinstance(x, np.ndarray) and x.ndim == 1,
        )
        catches = (ValueError, TypeError)
        self._test_robust_transformer(input_types, bad_input_checks, catches)

    def test_robust_transformer_sklearn(self):
        Transformers = (
            SimpleImputer,
            sklearn.preprocessing.StandardScaler,
            sklearn.preprocessing.Binarizer,
            sklearn.preprocessing.PolynomialFeatures,
        )
        # some of these input types are bad for sklearn.
        input_types = ('ser', 'df', 'arr1d')
        for Transformer in Transformers:
            robust_transformer = DelegatingRobustTransformer(Transformer())
            for input_type in input_types:
                X, y = self.d[input_type]
                robust_transformer.fit_transform(X, y=y)

    def _test_robust_transformer_pipeline(
        self, input_types, bad_input_checks, catches
    ):
        FragileTransformerPipeline3 = funcy.partial(
            FragileTransformerPipeline, 3)
        return self._test_robust_transformer(
            input_types,
            bad_input_checks,
            catches,
            transformer_maker=FragileTransformerPipeline3)

    def test_robust_transformer_pipeline_ser(self):
        input_types = ('ser',)
        bad_input_checks = (
            lambda x: isinstance(x, pd.Series),
        )
        catches = (ValueError, TypeError)
        self._test_robust_transformer_pipeline(
            input_types, bad_input_checks, catches)

    def test_robust_transformer_pipeline_df(self):
        input_types = ('ser', 'df',)
        bad_input_checks = (
            lambda x: isinstance(x, pd.Series),
            lambda x: isinstance(x, pd.DataFrame),
        )
        catches = (ValueError, TypeError)
        self._test_robust_transformer_pipeline(
            input_types, bad_input_checks, catches)

    def test_robust_transformer_pipeline_arr(self):
        input_types = ('ser', 'df', 'arr1d')
        bad_input_checks = (
            lambda x: isinstance(x, pd.Series),
            lambda x: isinstance(x, pd.DataFrame),
            lambda x: isinstance(x, np.ndarray) and x.ndim == 1,
        )
        catches = (ValueError, TypeError)
        self._test_robust_transformer_pipeline(
            input_types, bad_input_checks, catches)


class FeatureTest(unittest.TestCase):

    def setUp(self):
        self.input = 'foo'
        self.transformer = IdentityTransformer()

    def test_feature_init_with_one_transformer(self):
        Feature(self.input, self.transformer)

    def test_feature_init_with_list_of_transformer(self):
        Feature(self.input, [self.transformer, self.transformer])

    def test_feature_init_with_callable(self):
        Feature(self.input, funcy.identity)

    def test_feature_init_with_list_of_transformers_and_callables(self):
        Feature(self.input, [self.transformer, funcy.identity])

    def test_feature_init_invalid_transformer_api(self):
        with self.assertRaises(ValueError):
            Feature(self.input, object())

        with self.assertRaises(ValueError):
            Feature(self.input, IdentityTransformer)

    def test_feature_as_input_transformer_tuple(self):
        feature = Feature(self.input, self.transformer)
        tup = feature.as_input_transformer_tuple()
        self.assertIsInstance(tup, tuple)
        self.assertEqual(len(tup), 3)

    def test_feature_as_feature_engineering_pipeline(self):
        feature = Feature(self.input, self.transformer)
        mapper = feature.as_feature_engineering_pipeline()
        self.assertIsInstance(mapper, FeatureEngineeringPipeline)


class FeatureEngineeringPipelineTest(FeatureTest):

    def test_init_seqcont(self):
        feature = Feature(self.input, self.transformer)
        features = [feature]
        mapper = FeatureEngineeringPipeline(features)
        self.assertIsInstance(mapper, FeatureEngineeringPipeline)

    def test_init_scalar(self):
        feature = Feature(self.input, self.transformer)
        mapper = FeatureEngineeringPipeline(feature)
        self.assertIsInstance(mapper, FeatureEngineeringPipeline)

    def test_fit(self):
        feature = Feature(self.input, self.transformer)
        mapper = FeatureEngineeringPipeline(feature)
        df = pd.util.testing.makeCustomDataframe(5, 2)
        df.columns = ['foo', 'bar']
        mapper.fit(df)

    def test_transform(self):
        feature = Feature(self.input, self.transformer)
        mapper = FeatureEngineeringPipeline(feature)
        df = pd.util.testing.makeCustomDataframe(5, 2)
        df.columns = ['foo', 'bar']
        mapper.fit(df)
        X = mapper.transform(df)
        self.assertEqual(np.shape(X), (5, 1))
