import random

from ballet.util import asarray2d
from ballet.util.log import logger
from ballet.util.testing import seeded
from ballet.validation.base import FeaturePruner, FeaturePruningMixin
from ballet.validation.common import RandomFeaturePerformanceEvaluator
from ballet.validation.entropy import (
    estimate_conditional_information, estimate_entropy)
from ballet.validation.gfssf import (
    LAMBDA_1_ADJUSTMENT, LAMBDA_2_ADJUSTMENT, _compute_lmbdas,
    _compute_threshold, _concat_datasets)


class NoOpPruner(FeaturePruner):
    def prune(self):
        logger.info('Pruning features using {!s}'.format(self))
        return []


class RandomPruner(FeaturePruningMixin, RandomFeaturePerformanceEvaluator):

    def prune(self):
        """With probability p, select a random feature to prune"""
        logger.info('Pruning features using {!s}'.format(self))
        with seeded(self.seed):
            if random.uniform(0, 1) < self.p:
                return [random.choice(self.features)]


CMI_MESSAGE = "Calculating CMI of feature and target cond. on accpt features"


class GFSSFPruner(FeaturePruner):
    """A feature pruning evaluator that uses a modified version of
    GFSSF[1] - specifically, lines 12-13 of agGFSSF.

    Attributes:
        X_df (array-like): The dataset to build features off of.
        y (array-like): A single-column dataset representing the target
            feature.
        features (array-like): an array of ballet Features that have
            already been accepted, to be pruned.
        new_feature (ballet.Feature): the recently accepted Feature.
        lmbda_1 (float): A float used in GFSSF to calculate the information
            threshold. Default is a function of the entropy of y.
        lmbda_2 (float): A float used in GFSSF to calculate the information
            threshold. Default is a function of the entropy of y.

    References:
        [1] H. Li, X. Wu, Z. Li and W. Ding, "Group Feature Selection
            with Streaming Features," 2013 IEEE 13th International
            Conference on Data Mining, Dallas, TX, 2013, pp. 1109-1114.
            doi: 10.1109/ICDM.2013.137

    """

    def __init__(self, *args, lmbda_1=0.0, lmbda_2=0.0):
        super().__init__(*args)
        self.y = asarray2d(self.y)
        if lmbda_1 <= 0:
            lmbda_1 = estimate_entropy(self.y) / LAMBDA_1_ADJUSTMENT
        if lmbda_2 <= 0:
            lmbda_2 = estimate_entropy(self.y) / LAMBDA_2_ADJUSTMENT
        self.lmbda_1 = lmbda_1
        self.lmbda_2 = lmbda_2

    def prune(self):
        feature_dfs_by_src = {}
        for accepted_feature in [self.candidate_feature] + self.features:
            accepted_df = accepted_feature.as_feature_engineering_pipeline(
            ).fit_transform(self.X_df, self.y)
            feature_dfs_by_src[accepted_feature.source] = accepted_df

        lmbda_1, lmbda_2 = _compute_lmbdas(
            self.lmbda_1, self.lmbda_2, feature_dfs_by_src
        )

        logger.info(
            "Pruning features using GFSSF: lambda_1={l1}, lambda_2={l2}"
            .format(l1=lmbda_1, l2=lmbda_2)
        )

        redundant_features = []
        for candidate_feature in self.features:
            candidate_src = candidate_feature.source
            logger.debug("Pruning feature: {}".format(candidate_src))
            candidate_df = feature_dfs_by_src[candidate_src]
            _, n_candidate_cols = candidate_df.shape
            z = _concat_datasets(feature_dfs_by_src, omit=candidate_src)
            logger.debug(CMI_MESSAGE)
            cmi = estimate_conditional_information(candidate_df, self.y, z)

            logger.debug(
                "Conditional Mutual Information Score: {}".format(cmi))
            statistic = cmi
            threshold = _compute_threshold(lmbda_1, lmbda_2, n_candidate_cols)
            logger.debug("Calculated Threshold: {}".format(threshold))
            if statistic >= threshold:
                logger.debug(
                    "Passed, keeping feature: {}".format(candidate_src))
            else:
                logger.debug(
                    "Failed, found redundant feature: {}".format(candidate_src)
                )
                del feature_dfs_by_src[candidate_src]
                redundant_features.append(candidate_feature)
        return redundant_features
