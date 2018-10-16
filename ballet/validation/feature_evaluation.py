from ballet.validation.base import (
    FeatureAcceptanceEvaluator, FeaturePruningEvaluator)


class NoOpAcceptanceEvaluator(FeatureAcceptanceEvaluator):

    def judge(self, feature):
        return True


class NoOpPruningEvaluator(FeaturePruningEvaluator):

    def prune(self, k=4):
        return []
