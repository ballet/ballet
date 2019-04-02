from ballet.util import asarray2d
from ballet.util.log import logger
from ballet.validation.base import FeaturePruningEvaluator
from ballet.validation.entropy import (
    estimate_conditional_information, estimate_entropy)
from ballet.validation.gfssf import (
    LAMBDA_1_ADJUSTMENT, LAMBDA_2_ADJUSTMENT, _compute_lmbdas,
    _compute_threshold, _concat_datasets)


class NoOpPruningEvaluator(FeaturePruningEvaluator):

    def prune(self, k=4):
        return []


class GFSSFPruningEvaluator(FeaturePruningEvaluator):
    def __init__(self, X_df, y, features, new_feature, lmbda_1=0., lmbda_2=0.):
        super().__init__(X_df, y, features)
        self.y = asarray2d(y)
        if (lmbda_1 <= 0):
            lmbda_1 = estimate_entropy(self.y) / LAMBDA_1_ADJUSTMENT
        if (lmbda_2 <= 0):
            lmbda_2 = estimate_entropy(self.y) / LAMBDA_2_ADJUSTMENT
        self.lmbda_1 = lmbda_1
        self.lmbda_2 = lmbda_2
        self.feature = new_feature

    def prune(self):

        feature_dfs_by_src = {}
        for accepted_feature in [self.feature] + self.features:
            accepted_df = accepted_feature.as_dataframe_mapper().fit_transform(
                self.X_df, self.y)
            feature_dfs_by_src[accepted_feature.source] = accepted_df

        lmbda_1, lmbda_2 = _compute_lmbdas(
            self.lmbda_1, self.lmbda_2, feature_dfs_by_src)

        logger.info(
            'Prune Features using GFSSF: lambda_1={l1}, lambda_2={l2}'.format(
                l1=lmbda_1, l2=lmbda_2))
        redundant_features = []
        for candidate_feature in self.features:
            candidate_src = candidate_feature.source
            logger.debug(
                'Pruning feature: {}'.format(
                    candidate_src))
            candidate_df = feature_dfs_by_src[candidate_src]
            _, n_candidate_cols = candidate_df.shape
            z = _concat_datasets(feature_dfs_by_src, omit=candidate_src)
            cmi = estimate_conditional_information(candidate_df, self.y, z)
            logger.debug(
                'Conditional Mutual Information Score: {}'.format(cmi))
            statistic = cmi
            threshold = _compute_threshold(
                lmbda_1, lmbda_2, n_candidate_cols)
            logger.debug('Calculated Threshold: {}'.format(threshold))
            if statistic >= threshold:
                logger.debug(
                    'Passed, keeping feature: {}'.format(candidate_src)
                )
            else:
                logger.debug(
                    'Failed, found redundant feature: {}'.format(candidate_src)
                )
                del feature_dfs_by_src[candidate_src]
                redundant_features.append(candidate_feature)
        return redundant_features
