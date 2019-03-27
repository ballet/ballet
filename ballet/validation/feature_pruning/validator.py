class NoOpPruningEvaluator(FeaturePruningEvaluator):

    def prune(self, k=4):
        return []
