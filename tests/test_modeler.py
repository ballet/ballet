import unittest

import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer

from ballet.modeling.problem import MulticlassClassificationProblem, RegressionProblem
from ballet.modeler import (
    DecisionTreeModeler, StratifiedKFoldMultiClassIndicator, TunedModeler,
    TunedRandomForestClassifier, TunedRandomForestRegressor)
from ballet.util.log import logger
from ballet.util.testing import EPSILON, log_seed_on_error, seeded


class _CommonTesting:

    def setUp(self):
        # Create fake data
        X_classification, y_classification = sklearn.datasets.load_iris(
            return_X_y=True)
        X_regression, y_regression = sklearn.datasets.load_boston(
            return_X_y=True)

        # TODO: add test for BinaryClassificationProblem
        self.data = {
            MulticlassClassificationProblem: {
                "X": X_classification,
                "y": y_classification,
            },
            RegressionProblem: {
                "X": X_regression,
                "y": y_regression,
            },
        }

        self.data_pd = {
            MulticlassClassificationProblem: {
                "X": pd.DataFrame(X_classification),
                "y": pd.DataFrame(y_classification),
            },
            RegressionProblem: {
                "X": pd.DataFrame(X_regression),
                "y": pd.DataFrame(y_regression),
            },
        }

    def _setup_modeler(self, problem_type, data):
        X = data[problem_type]["X"]
        y = data[problem_type]["y"]
        modeler = self.modeler_class(problem_type=problem_type)
        return modeler, X, y

    def _test_problem_type_cv(self, problem_type, data):
        modeler, X, y = self._setup_modeler(problem_type, data)
        metrics = modeler.compute_metrics_cv(X, y)
        return metrics

    def _test_problem_type_train_test(self, problem_type, data):
        modeler, X, y = self._setup_modeler(problem_type, data)
        n = round(0.7 * len(X))
        metrics = modeler.compute_metrics_train_test(X, y, n=n)

        return metrics

    def _call_method(self, method, problem_type, seed=None):
        with seeded(seed):
            metrics = getattr(self, method)(problem_type, self.data)
        with seeded(seed):
            metrics_pd = getattr(self, method)(problem_type, self.data_pd)
        return metrics, metrics_pd

    def _prepare_metrics_for_assertions(self, metrics):
        return {
            metric['description']: metric['value']
            for metric in metrics
        }


class TestModeler(_CommonTesting, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.modeler_class = DecisionTreeModeler

    def test_classification_cv(self):
        metrics, metrics_pd = self._call_method(
            '_test_problem_type_cv', MulticlassClassificationProblem, seed=17)
        self.assertEqual(metrics, metrics_pd)
        metrics = self._prepare_metrics_for_assertions(metrics)
        self.assertAlmostEqual(
            metrics['Accuracy'], 0.9, delta=0.1 - EPSILON)
        self.assertAlmostEqual(
            metrics['ROC AUC Score'], 0.9, delta=0.1 - EPSILON)

        # todo multiclass
        # self.assertAlmostEqual(
        #     metrics['Precision'], 0.9403594, delta=EPSILON)
        # self.assertAlmostEqual(
        #     metrics['Recall'], 0.9403594, delta=EPSILON)

    def test_classification_train_test(self):
        metrics, metrics_pd = self._call_method(
            '_test_problem_type_train_test', MulticlassClassificationProblem,
            seed=1093)
        self.assertEqual(metrics, metrics_pd)
        metrics = self._prepare_metrics_for_assertions(metrics)
        self.assertAlmostEqual(
            metrics['Accuracy'], 0.9, delta=0.1 - EPSILON)
        self.assertAlmostEqual(
            metrics['ROC AUC Score'], 0.9, delta=0.1 - EPSILON)

        # # todo multiclass
        # self.assertAlmostEqual(
        #     metrics['Precision'], 0.7777777, delta=EPSILON)
        # self.assertAlmostEqual(
        #     metrics['Recall'], 0.7777777, delta=EPSILON)

    def test_regression_cv(self):
        metrics, metrics_pd = self._call_method(
            '_test_problem_type_cv', RegressionProblem)
        self.assertEqual(metrics, metrics_pd)
        metrics = self._prepare_metrics_for_assertions(metrics)
        self.assertAlmostEqual(
            metrics['Negative Mean Squared Error'], -20.7, delta=0.1 - EPSILON)
        self.assertAlmostEqual(
            metrics['R-squared'], 0.7, delta=0.1 - EPSILON)

    def test_regression_train_test(self):
        metrics, metrics_pd = self._call_method(
            '_test_problem_type_train_test', RegressionProblem, seed=4)
        self.assertEqual(metrics, metrics_pd)
        metrics = self._prepare_metrics_for_assertions(metrics)
        self.assertAlmostEqual(
            metrics['Negative Mean Squared Error'], -34.7, delta=0.1 - EPSILON)
        self.assertAlmostEqual(
            metrics['R-squared'], 0.6, delta=0.1 - EPSILON)


class TestTunedModelers(_CommonTesting, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.modeler_class = TunedModeler

    def _setup_modeler(self, problem_type, data):
        modeler, X, y = super()._setup_modeler(problem_type, data)
        return modeler, X, y

    def test_classification_cv(self):
        metrics, metrics_pd = self._call_method(
            '_test_problem_type_cv', MulticlassClassificationProblem, seed=1)
        self.assertEqual(metrics, metrics_pd)

    def test_classification_train_test(self):
        metrics, metrics_pd = self._call_method(
            '_test_problem_type_train_test', MulticlassClassificationProblem,
            seed=2)
        self.assertEqual(metrics, metrics_pd)

    def test_regression_cv(self):
        metrics, metrics_pd = self._call_method(
            '_test_problem_type_cv', RegressionProblem, seed=3)

    def test_regression_train_test(self):
        metrics, metrics_pd = self._call_method(
            '_test_problem_type_train_test', RegressionProblem, seed=4)
        self.assertEqual(metrics, metrics_pd)

    def _test_tuned_random_forest_estimator(self, Estimator, problem_type):
        model = Estimator()
        data = self.data[problem_type]
        X, y = data['X'], data['y']
        with log_seed_on_error(logger):
            model.fit(X, y, tune=False)
            old_score = model.score(X, y)
            model.fit(X, y, tune=True)
            new_score = model.score(X, y)
            self.assertGreaterEqual(new_score, old_score)

    def test_tuned_random_forest_regressor(self):
        self._test_tuned_random_forest_estimator(
            TunedRandomForestRegressor, RegressionProblem)

    def test_tuned_random_forest_classifier(self):
        self._test_tuned_random_forest_estimator(
            TunedRandomForestClassifier, MulticlassClassificationProblem)


class TestStratifiedKFoldMultiClassIndicator(unittest.TestCase):
    def test(self):
        X, y = sklearn.datasets.load_iris(return_X_y=True)
        transformer = LabelBinarizer()
        ym = transformer.fit_transform(y)

        params = {'random_state': 1, 'n_splits': 3}
        kf = StratifiedKFold(**params)
        kfm = StratifiedKFoldMultiClassIndicator(transformer, **params)

        kf_folds = kf.split(X, y)
        kfm_folds = kfm.split(X, ym)

        for (inds_tr_kf, inds_te_kf), (inds_tr_kfm, inds_te_kfm) \
                in zip(kf_folds, kfm_folds):
            self.assertTrue(np.array_equal(inds_tr_kf, inds_tr_kfm))
            self.assertTrue(np.array_equal(inds_te_kf, inds_te_kfm))
