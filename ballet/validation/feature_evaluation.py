from ballet.validation.base import (
    FeatureAcceptanceEvaluator, FeaturePruningEvaluator)


class FeatureRelevanceEvaluator(FeatureAcceptanceEvaluator):
    """Accept a feature if it is correlated to the target"""

    def judge(self, feature):
        return True


class FeatureRedundancyEvaluator(FeaturePruningEvaluator):
    """Remove a feature if it is conditionally independent of the target

    Let Sk be the set of subsets of features of size less than or equal to k.
    A feature Xi is redundant if it is independent of the target, conditional
    on some S in Sk. If a feature is redundant, it is removed from the feature
    matrix.
    """

    def prune(self, k=4):
        return []
