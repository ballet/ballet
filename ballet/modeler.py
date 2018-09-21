from abc import ABCMeta, abstractmethod
import os

import btb
import btb.tuning.gp
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.externals import joblib
from sklearn.model_selection import (
    KFold, StratifiedKFold, cross_val_score, cross_validate, train_test_split,
)
from sklearn.model_selection._validation import _multimetric_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ballet.modeling.constants import RANDOM_STATE
from ballet.modeling.io_transformers import FeatureTypeTransformer, TargetTypeTransformer
from ballet.modeling.scoring import ScorerInfo, get_scorer_names_for_problem_type
from ballet.util.log import logger


class Modeler:
    '''Versatile modeling object.

    Handles classification and regression problems and computes variety of
    performance metrics.

    Args:
        problem_type (ProblemType)
    '''

    def __init__(self, problem_type=None, scorers=None):
        self.problem_type = problem_type

        if scorers is None:
            names = get_scorer_names_for_problem_type(self.problem_type)
        self.scorers_info = [
            ScorerInfo(name=name)
            for name in names
        ]
        self.scorers = [
            scorer_info.scorer
            for scorer_info in self.scorers_info
        ]
        self.primary_scorer = self.scorers[0]

        self.estimator = self._get_default_estimator()
        self.feature_type_transformer = FeatureTypeTransformer()

        needs_label_binarizer = self.problem_type.multi_classification
        self.target_type_transformer = TargetTypeTransformer(
            needs_label_binarizer=needs_label_binarizer)

    def set_estimator(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, **kwargs):
        X, y = self._format_inputs(X, y)
        self.estimator.fit(X, y, **kwargs)

    def predict(self, X):
        X = self._format_X(X)
        return self.estimator.predict(X)

    def predict_proba(self, X):
        X = self._format_X(X)
        return self.estimator.predict_proba(X)

    def score(self, X, y):
        X, y = self._format_inputs(X, y)
        return self.estimator.score(X, y)

    def dump(self, filepath):
        joblib.dump(self.estimator, filepath, compress=True)

    def load(self, filepath):
        if not os.path.exists(filepath):
            raise ValueError("Couldn't find model at {}".format(filepath))
        self.estimator = joblib.load(filepath)

    def compute_metrics_cv(self, X, y, **kwargs):
        '''Compute cross-validated metrics.

        Trains this model on data X with labels y.
        Returns a list of dict with keys name, scoring_name, value.

        Args:
        X (Union[np.array, pd.DataFrame]): data
        y (Union[np.array, pd.DataFrame, pd.Series]): labels
        '''

        # compute scores
        results = self.cv_score_mean(X, y)
        return results

    def _compute_metrics_train_test(self, X_tr, y_tr, X_te, y_te):
        '''Compute metrics on test set, given entire train-test split'''
        X_tr, y_tr = self._format_inputs(X_tr, y_tr)
        X_te, y_te = self._format_inputs(X_te, y_te)

        # fit model on entire training set
        self.estimator.fit(X_tr, y_tr)

        scorers = {
            scorer_info.name: scorer_info.scorer
            for scorer_info in self.scorers_info
        }
        multimetric_score_results = _multimetric_score(
            self.estimator, X_te, y_te, scorers)

        return self._process_cv_results(
            multimetric_score_results, filter_testing_keys=False)

    def compute_metrics_train_test(self, X, y, n):
        '''Compute metrics on test set, doing train-test split on inputs'''
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, train_size=n, test_size=len(y) - n, shuffle=True)
        return self._compute_metrics_train_test(X_tr, y_tr, X_te, y_te)

    def cv_score_mean(self, X, y):
        '''Compute mean score across cross validation folds.

        Split data and labels into cross validation folds and fit the model for
        each fold. Then, for each scoring type in scorings, compute the score.
        Finally, average the scores across folds. Returns a dictionary mapping
        scoring to score.

        Args:
        X (np.array): data
        y (np.array): labels
        scorings (List[str]): scoring types
        '''

        X, y = self._format_inputs(X, y)

        if self.problem_type.binary_classification:
            kf = StratifiedKFold(
                shuffle=True, random_state=RANDOM_STATE + 3)
        elif self.problem_type.multi_classification:
            self.target_type_transformer.inverse_transform(y)
            transformer = self.target_type_transformer
            kf = StratifiedKFoldMultiClassIndicator(
                transformer, shuffle=True, n_splits=3, random_state=RANDOM_STATE + 3)
        elif self.problem_type.regression:
            kf = KFold(shuffle=True, n_splits=3, random_state=RANDOM_STATE + 4)
        else:
            raise NotImplementedError

        scoring = {
            scorer_info.name: scorer_info.scorer
            for scorer_info in self.scorers_info
        }
        cv_results = cross_validate(
            self.estimator, X, y,
            scoring=scoring, cv=kf, return_train_score=False)

        # post-processing
        results = self._process_cv_results(cv_results)
        return results

    def _process_cv_results(self, cv_results, filter_testing_keys=True):
        result = []
        for key, val in cv_results.items():
            if filter_testing_keys:
                if key.startswith('test_'):
                    scorer_name = key[len('test_'):]
                else:
                    continue
            else:
                scorer_name = key
            scorer_description = ScorerInfo(name=scorer_name).description
            val = np.nanmean(cv_results[key])
            if np.isnan(val):
                val = None
            result.append({
                'name': scorer_name,
                'description': scorer_description,
                'value': val,
            })

        return result

    def _format_inputs(self, X, y):
        return self._format_X(X), self._format_y(y)

    def _format_y(self, y):
        return self.target_type_transformer.fit_transform(y)

    def _format_X(self, X):
        return self.feature_type_transformer.fit_transform(X)

    def _get_default_estimator(self):
        if self.problem_type.classification:
            return self._get_default_classifier()
        elif self.problem_type.regression:
            return self._get_default_regressor()
        else:
            raise NotImplementedError

    def _get_default_classifier(self):
        return RandomForestClassifier(n_estimators=10, random_state=RANDOM_STATE + 1)

    def _get_default_regressor(self):
        return RandomForestRegressor(n_estimators=10, random_state=RANDOM_STATE + 2)


class DecisionTreeModeler(Modeler):

    def _get_default_classifier(self):
        return DecisionTreeClassifier(random_state=RANDOM_STATE + 1)

    def _get_default_regressor(self):
        return DecisionTreeRegressor(random_state=RANDOM_STATE + 2)


class SelfTuningMixin(metaclass=ABCMeta):

    @abstractmethod
    def get_tunables(self):
        return None

    @property
    def tunables(self):
        if not hasattr(self, '_tunables'):
            self._tunables = self.get_tunables()
        return self._tunables

    @tunables.setter
    def tunables(self, tunables):
        self._tunables = tunables

    @property
    def tuning_cv(self):
        if not hasattr(self, '_tuning_cv'):
            self._tuning_cv = 3
        return self._tuning_cv

    @tuning_cv.setter
    def tuning_cv(self, tuning_cv):
        self._tuning_cv = tuning_cv

    @property
    def tuning_iter(self):
        if not hasattr(self, '_tuning_iter'):
            self._tuning_iter = 3
        return self._tuning_iter

    @tuning_iter.setter
    def tuning_iter(self, tuning_iter):
        self._tuning_iter = tuning_iter

    def _get_parent_instance(self):
        # this is probably a sign of bad design pattern
        mro = type(self).mro()
        ParentClass = mro[mro.index(__class__) + 1]  # noqa
        return ParentClass()

    def fit(self, X, y, tune=True, **fit_kwargs):
        if tune:
            # do some tuning
            if self.tunables is not None:

                scorer = None
                def score(estimator):
                    scores = cross_val_score(
                        estimator, X, y,
                        scoring=scorer, cv=self.tuning_cv,
                        fit_params=fit_kwargs)
                    return np.mean(scores)

                logger.info('Tuning model using BTB GP tuner...')
                tuner = btb.tuning.gp.GP(self.tunables)
                estimator = self._get_parent_instance()
                original_score = score(estimator)
                # TODO: this leads to an error because default value of
                # max_depth for RF is `None`
                # params = funcy.project(
                #     estimator.get_params(), [t[0] for t in self.tunables])
                # tuner.add(params, original_score)
                for i in range(self.tuning_iter):
                    params = tuner.propose()
                    estimator.set_params(**params)
                    score_ = score(estimator)
                    logger.debug(
                        'Iteration {}, params {}, score {}'
                        .format(i, params, score_)
                    )
                    tuner.add(params, score_)

                best_params = tuner._best_hyperparams
                best_score = tuner._best_score
                self.set_params(**best_params)
                logger.info(
                    'Tuning complete. '
                    'Cross val score changed from {:0.3f} to {:0.3f}.'
                    .format(original_score, best_score))
            else:
                logger.warning('Tuning requested, but either btb not '
                                'installed or tunable HyperParameters not '
                                'specified.')

        return super().fit(X, y, **fit_kwargs)


class SelfTuningRandomForestMixin(SelfTuningMixin):

    def get_tunables(self):
        return [
            ('n_estimators',
             btb.HyperParameter(btb.ParamTypes.INT, [10, 500])),
            ('max_depth',
             btb.HyperParameter(btb.ParamTypes.INT, [3, 20]))
        ]


class TunedRandomForestRegressor(SelfTuningRandomForestMixin, RandomForestRegressor):
    pass


class TunedRandomForestClassifier(SelfTuningRandomForestMixin, RandomForestClassifier):
    pass


class TunedModeler(Modeler):

    def _get_default_classifier(self):
        return TunedRandomForestClassifier(random_state=RANDOM_STATE + 1)

    def _get_default_regressor(self):
        return TunedRandomForestRegressor(random_state=RANDOM_STATE + 2)


class StratifiedKFoldMultiClassIndicator(StratifiedKFold):
    '''Adaptation of StratifiedKFold to support multiclass-indicator format y values.

    Note that this should not be used for multilabel, multiclass dataself.
    '''

    def __init__(self, transformer, *args, **kwargs):
        self.transformer = transformer
        super().__init__(*args, **kwargs)

    def split(self, X, y, groups=None):
        y = self.transformer.inverse_transform(y)
        return super().split(X, y, groups=None)
