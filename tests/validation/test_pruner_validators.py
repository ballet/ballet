import unittest

import numpy as np

from ballet import Feature
from ballet.eng.base import SimpleFunctionTransformer
from ballet.eng.misc import IdentityTransformer
from ballet.util import asarray2d
from ballet.validation.feature_pruning.validator import GFSSFPruningEvaluator

from tests.util import load_regression_data


class GFSSFPrunerTest(unittest.TestCase):

    def setUp(self):
        self.X, self.y = load_regression_data(n_informative=1,
                                              n_uninformative=14)

    def test_prune_exact_replicas(self):
        feature_1 = Feature(
            input='A_0',
            transformer=IdentityTransformer(),
            source='1st Feature')
        feature_2 = Feature(
            input='A_0',
            transformer=IdentityTransformer(),
            source='2nd Feature')
        gfssf_pruner = GFSSFPruningEvaluator(
            self.X, self.y, [feature_1], feature_2)

        redunant_features = gfssf_pruner.prune()
        self.assertIn(
            feature_1,
            redunant_features,
            'Exact replica features should be pruned')

    @unittest.expectedFailure
    def test_prune_weak_replicas(self):
        def add_noise(X):
            X = asarray2d(X)
            return X + np.random.normal(0, 0.5, X.shape)

        noise_transformer = SimpleFunctionTransformer(add_noise)
        feature_weak = Feature(
            input='A_0',
            transformer=noise_transformer,
            source='1st Feature')
        feature_strong = Feature(
            input='A_0',
            transformer=IdentityTransformer(),
            source='2nd Feature')
        gfssf_pruner = GFSSFPruningEvaluator(
            self.X, self.y, [feature_weak], feature_strong)

        redunant_features = gfssf_pruner.prune()
        self.assertIn(
            feature_weak,
            redunant_features,
            'Noisy features should be pruned')

    def test_prune_keep_relevant(self):
        feature_1 = Feature(
            input='A_0',
            transformer=IdentityTransformer(),
            source='1st Feature')
        feature_2 = Feature(
            input='Z_0',
            transformer=IdentityTransformer(),
            source='2nd Feature')
        gfssf_pruner = GFSSFPruningEvaluator(
            self.X, self.y, [feature_1], feature_2)

        redunant_features = gfssf_pruner.prune()
        self.assertNotIn(
            feature_1,
            redunant_features,
            'Still relevant features should be pruned')

    @unittest.expectedFailure
    def test_prune_irrelevant_features(self):
        feature_1 = Feature(
            input='Z_0',
            transformer=IdentityTransformer(),
            source='1st Feature')
        feature_2 = Feature(
            input='A_0',
            transformer=IdentityTransformer(),
            source='2nd Feature')
        gfssf_pruner = GFSSFPruningEvaluator(
            self.X, self.y, [feature_1], feature_2)

        redunant_features = gfssf_pruner.prune()
        self.assertIn(
            feature_1,
            redunant_features,
            'Irrelevant features should be pruned')
