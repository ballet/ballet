from abc import ABCMeta, abstractmethod

from funcy import constantly, ignore, post_processing

from ballet.util import whether_failures


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

    def __init__(self, X_df, y, features):
        self.X_df = X_df
        self.y = y
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


class BaseCheck(metaclass=ABCMeta):

    @ignore(Exception, default=False)
    @post_processing(constantly(True))
    def do_check(self, obj):
        return self.check(obj)

    @abstractmethod
    def check(self, obj):
        pass


@whether_failures
def check_from_class(check_class, obj, *checker_args, **checker_kwargs):
    for Checker in check_class.__subclasses__():
        check = Checker(*checker_args, **checker_kwargs).do_check
        name = Checker.__name__
        success = check(obj)
        if not success:
            yield name
