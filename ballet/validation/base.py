from abc import ABCMeta, abstractmethod

from funcy import constantly, ignore, post_processing


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

    def __init__(self, X_df, y, features, candidate_feature):
        self.X_df = X_df
        self.y = y
        self.features = features
        self.candidate_feature = candidate_feature

    def __str__(self):
        return self.__class__.__name__


class FeatureAcceptanceMixin(metaclass=ABCMeta):

    @abstractmethod
    def judge(self):
        """Judge whether feature should be accepted

        Returns:
            bool: feature should be accepted
        """
        pass


class FeaturePruningMixin(metaclass=ABCMeta):

    @abstractmethod
    def prune(self):
        """Prune existing features

        Returns:
            list: list of features to remove
        """
        pass


class FeatureAccepter(FeatureAcceptanceMixin, FeaturePerformanceEvaluator):
    """Accept/reject a feature to the project based on its performance"""
    pass


class FeaturePruner(FeaturePruningMixin, FeaturePerformanceEvaluator):
    """Prune features after acceptance based on their performance"""
    pass


class BaseCheck(metaclass=ABCMeta):

    @ignore(Exception, default=False)
    @post_processing(constantly(True))
    def do_check(self, obj):
        return self.check(obj)

    @abstractmethod
    def check(self, obj):
        pass
