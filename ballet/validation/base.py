from abc import ABCMeta, abstractmethod

from funcy import constantly, ignore, post_processing

from ballet.util import get_subclasses


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

    def do_all_checks(self, item):
        """Check an item according to checker subclasses

        For all checker subclasses (subclasses of self's class), instantiates
        the checker class, optionally with the provided args and kwargs. Then
        checks the item.

        Args:
            item: item to check

        Returns:
            Dict[str, bool]: mapping from check names to check outcomes
        """
        result = {}
        for Checker in get_subclasses(type(self)):
            check = Checker.check
            name = Checker.__name__
            outcome = check(self, item)
            result[name] = bool(outcome)
        return result

    @ignore(Exception, default=False)
    @post_processing(constantly(True))
    def do_check(self, item):
        return self.check(item)

    @abstractmethod
    def check(self, item):
        pass
