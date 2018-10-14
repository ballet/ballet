from abc import ABCMeta, abstractmethod


class BaseValidator(metaclass=ABCMeta):
    """Base class for a generic validator"""

    @abstractmethod
    def validate(self):
        """Validate something

        Returns:
            bool: validation succeeded
        """
        pass


class FeaturePerformanceEvaluator(metaclass=ABCMeta):
    """Evaluate the performance of features from an ML point-of-view"""

    def __init__(self, X_df, y_df, features):
        self.X_df = X_df
        self.y_df = y_df
        self.features = features


class FeatureAcceptanceEvaluator(FeaturePerformanceEvaluator):
    """Accept/reject a feature to the project based on its performance"""

    @abstractmethod
    def judge(self, feature):
        """Judge whether feature should be accepted

        Returns:
            bool: feature should be accepted
        """
        pass


class FeaturePruningEvaluator(FeaturePerformanceEvaluator):
    """Prune features after acceptance based on their performance"""

    @abstractmethod
    def prune(self):
        """Prune existing features

        Returns:
            list: list of features to remove
        """
        pass
