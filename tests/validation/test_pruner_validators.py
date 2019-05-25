import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

from ballet import Feature
from ballet.eng.base import SimpleFunctionTransformer
from ballet.eng.misc import IdentityTransformer
from ballet.util import asarray2d
from ballet.validation.feature_pruning.validator import GFSSFPruningEvaluator


class GFSSFPrunerTest(unittest.TestCase):
    def setUp(self):
        p = 15
        q = 1
        X, y, coef = make_regression(
            n_samples=500, n_features=p, n_informative=q, coef=True,
            shuffle=True, random_state=1)

        # informative columns are 'A', 'B'
        # uninformative columns are 'Z_0', ..., 'Z_11'
        columns = []
        informative = list('A')
        other = ['Z_{i}'.format(i=i) for i in reversed(range(p - q))]
        for i in range(p):
            if coef[i] == 0:
                columns.append(other.pop())
            else:
                columns.append(informative.pop())

        self.X = pd.DataFrame(data=X, columns=columns)
        self.y = pd.Series(y)

    def test_prune_exact_replicas(self):
        feature_1 = Feature(
            input='A',
            transformer=IdentityTransformer(),
            source='1st Feature')
        feature_2 = Feature(
            input='A',
            transformer=IdentityTransformer(),
            source='2nd Feature')
        gfssf_pruner = GFSSFPruningEvaluator(
            self.X, self.y, [feature_1], feature_2)

        redunant_features = gfssf_pruner.prune()
        self.assertIn(
            feature_1,
            redunant_features,
            'Exact replica features should be pruned')

    @unittest.skip
    def test_prune_weak_replicas(self):
        def add_noise(X):
            X = asarray2d(X)
            return X + np.random.normal(0, 0.5, X.shape)

        noise_transformer = SimpleFunctionTransformer(add_noise)
        feature_weak = Feature(
            input='A',
            transformer=noise_transformer,
            source='1st Feature')
        feature_strong = Feature(
            input='A',
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
            input='A',
            transformer=IdentityTransformer(),
            source='1st Feature')
        feature_2 = Feature(
            input='Z_1',
            transformer=IdentityTransformer(),
            source='2nd Feature')
        gfssf_pruner = GFSSFPruningEvaluator(
            self.X, self.y, [feature_1], feature_2)

        redunant_features = gfssf_pruner.prune()
        self.assertNotIn(
            feature_1,
            redunant_features,
            'Still relevant features should be pruned')

    def test_prune_irrelevant_features(self):
        feature_1 = Feature(
            input='Z_1',
            transformer=IdentityTransformer(),
            source='1st Feature')
        feature_2 = Feature(
            input='A',
            transformer=IdentityTransformer(),
            source='2nd Feature')
        gfssf_pruner = GFSSFPruningEvaluator(
            self.X, self.y, [feature_1], feature_2)

        redunant_features = gfssf_pruner.prune()
        self.assertIn(
            feature_1,
            redunant_features,
            'Irrelevant features should be pruned')
