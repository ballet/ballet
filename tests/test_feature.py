import unittest

import funcy
import numpy as np
import pandas as pd

from ballet.eng.misc import IdentityTransformer
from ballet.feature import Feature
from ballet.pipeline import FeatureEngineeringPipeline


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

    def test_feature_init_with_none(self):
        Feature(self.input, None)

    def test_feature_init_with_list_of_none(self):
        Feature(self.input, [None, None])

    def test_feature_init_with_list_of_none_and_notnone(self):
        Feature(self.input, [None, self.transformer])

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
