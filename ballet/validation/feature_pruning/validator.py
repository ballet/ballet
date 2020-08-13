import random

from ballet.util.log import logger
from ballet.util.testing import seeded
from ballet.validation.base import FeaturePruner, FeaturePruningMixin
from ballet.validation.common import RandomFeaturePerformanceEvaluator
from ballet.validation.entropy import estimate_conditional_information
from ballet.validation.gfssf import (
    GFSSFPerformanceEvaluator, _compute_lmbdas, _compute_threshold,
    _concat_datasets)


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


class GFSSFPruner(FeaturePruningMixin, GFSSFPerformanceEvaluator):

    def prune(self):
        """Prune using GFSSF

        Uses lines 12-13 of agGFSSF
        """

        logger.info(f'Pruning features using {self}')

        feature_dfs_by_src = self._get_feature_dfs_by_src()
        lmbda_1, lmbda_2 = _compute_lmbdas(
            self.lmbda_1, self.lmbda_2, feature_dfs_by_src
        )

        logger.info(f'Recomputed lambda_1={lmbda_1}, lambda_2={lmbda_2}')

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
