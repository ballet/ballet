import numpy as np

from ballet.util import asarray2d
from ballet.util.log import logger
from ballet.validation.base import FeatureAcceptanceEvaluator
from ballet.validation.entropy import (
    estimate_conditional_information, estimate_entropy)

LAMBDA_1_ADJUSTMENT = 32
LAMBDA_2_ADJUSTMENT = 32


def _concat_datasets(dfs_by_src, n_samples, omit=None):
    filtered_dfs = [np.array(dfs_by_src[x])
                    for x in dfs_by_src if x is not omit]
    return asarray2d(np.concatenate(filtered_dfs, axis=1))


def _compute_lmbdas(unnorm_lmbda_1, unnorm_lmbda_2, acc_by_src):
    feat_srcs = acc_by_src.keys()
    num_features = len(feat_srcs)
    num_feature_cols = 0
    for acc_feat_src in feat_srcs:
        num_feature_cols += acc_by_src[acc_feat_src].shape[1]
    return (unnorm_lmbda_1 / num_features, unnorm_lmbda_2 / num_feature_cols)


def _compute_threshold(lmbda_1, lmbda_2, n_feature_cols):
    return lmbda_1 + lmbda_2 * n_feature_cols


class GFSSFPruningEvaluator(FeatureAcceptanceEvaluator):
    def __init__(self, X_df, y, features, lmbda_1=0., lmbda_2=0.):
        super().__init__(X_df, y, features)
        self.y = asarray2d(y)
        if (lmbda_1 <= 0):
            lmbda_1 = estimate_entropy(self.y) / LAMBDA_1_ADJUSTMENT
        if (lmbda_2 <= 0):
            lmbda_2 = estimate_entropy(self.y) / LAMBDA_2_ADJUSTMENT
        self.lmbda_1 = lmbda_1
        self.lmbda_2 = lmbda_2

    def prune(self, feature):

        feature_dfs_by_src = {}
        for accepted_feature in [feature] + self.features:
            accepted_df = accepted_feature.as_dataframe_mapper().fit_transform(
                self.X_df, self.y)
            feature_dfs_by_src[accepted_feature.source] = accepted_df

        lmbda_1, lmbda_2 = _compute_lmbdas(
            self.lmbda_1, self.lmbda_2, feature_dfs_by_src)

        logger.info(
            'Judging Feature using GFSSF: lambda_1={l1}, lambda_2={l2}'.format(
                l1=lmbda_1, l2=lmbda_2))
        redundant_features = []
        for candidate_feature in self.features:
            candidate_src = candidate_feature.source
            logger.debug(
                'Judging feature: {}'.format(
                    candidate_src))
            candidate_df = feature_dfs_by_src[candidate_src]
            _, n_candidate_cols = candidate_df.shape
            z = _concat_datasets(feature_dfs_by_src, candidate_src)
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
