from abc import ABCMeta, abstractmethod
from typing import Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from ballet.feature import Feature


class BaseValidator(metaclass=ABCMeta):
    """Base class for a generic validator"""

    @abstractmethod
    def validate(self) -> bool:
        """Validate something and return whether validation succeeded"""
        pass


class FeaturePerformanceEvaluator(metaclass=ABCMeta):
    """Evaluate the performance of features from an ML point-of-view

    Args:
        X_df_fit: entities frame for fitting the features
        y_df_fit: targets frame/series for fitting the features
        X_df_eval: entities frame for evaluating the features
        y_eval: target values for evaluating the features
    """

    def __init__(self,
                 X_df_fit: pd.DataFrame,
                 y_df_fit: Union[pd.DataFrame, pd.Series],
                 X_df_eval: pd.DataFrame,
                 y_eval: np.ndarray,
                 features: Iterable[Feature],
                 candidate_feature: Feature):
        self.X_df = X_df_fit
        self.y_df = y_df_fit
        self.y = y_eval
        self.features = features
        self.candidate_feature = candidate_feature

    def __str__(self):
        return self.__class__.__name__


class FeatureAcceptanceMixin(metaclass=ABCMeta):

    @abstractmethod
    def judge(self) -> bool:
        """Judge whether feature should be accepted"""
        pass


class FeaturePruningMixin(metaclass=ABCMeta):

    @abstractmethod
    def prune(self) -> List[Feature]:
        """Prune existing features, returning list of features to remove"""
        pass


class FeatureAccepter(FeatureAcceptanceMixin, FeaturePerformanceEvaluator):
    """Accept/reject a feature to the project based on its performance"""
    pass


class FeaturePruner(FeaturePruningMixin, FeaturePerformanceEvaluator):
    """Prune features after acceptance based on their performance"""
    pass


class BaseCheck(metaclass=ABCMeta):

    def do_check(self, obj) -> bool:
        """Do the check and return whether an exception was *not* thrown"""
        try:
            self.check(obj)
        except Exception:
            return False
        else:
            return True

    @abstractmethod
    def check(self, obj) -> None:
        """Check something and throw an exception if the thing is bad"""
        pass

    def give_advice(self, feature) -> Optional[str]:
        """Description of how to resolve if the check fails"""
        return None
