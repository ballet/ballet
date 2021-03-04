import random

from ballet.util.log import logger
from ballet.util.testing import seeded
from ballet.validation.base import FeaturePruner, FeaturePruningMixin
from ballet.validation.common import RandomFeaturePerformanceEvaluator
from ballet.validation.entropy import estimate_conditional_information
from ballet.validation.gfssf import (
    GFSSFPerformanceEvaluator, _compute_lmbdas, _compute_threshold,
    _concat_datasets,)


class NoOpPruner(FeaturePruner):
    def prune(self):
        logger.info(f'Pruning features using {self}')
        return []


class RandomPruner(FeaturePruningMixin, RandomFeaturePerformanceEvaluator):

    def prune(self):
        """With probability p, select a random feature to prune"""
        logger.info(f'Pruning features using {self}')
        with seeded(self.seed):
            if random.uniform(0, 1) < self.p:
                return [random.choice(self.features)]


CMI_MESSAGE = 'Calculating CMI of feature and target cond. on accpt features'


class GFSSFPruner(FeaturePruningMixin, GFSSFPerformanceEvaluator):

    def prune(self):
        """Prune using GFSSF

        Uses lines 12-13 of agGFSSF
        """

        logger.info(f'Pruning features using {self}')

        feature_df_map = self._get_feature_df_map()
        lmbda_1, lmbda_2 = _compute_lmbdas(
            self.lmbda_1, self.lmbda_2, feature_df_map)

        logger.info(f'Recomputed lambda_1={lmbda_1}, lambda_2={lmbda_2}')

        redundant_features = []
        for candidate_feature in self.features:
            candidate_src = candidate_feature.source
            logger.debug(
                f'Trying to prune feature with source {candidate_src}')
            candidate_df = feature_df_map[candidate_feature]
            _, n_candidate_cols = candidate_df.shape
            z = _concat_datasets(feature_df_map, omit=[candidate_feature])
            logger.debug(CMI_MESSAGE)
            cmi = estimate_conditional_information(candidate_df, self.y, z)

            logger.debug(f'Conditional Mutual Information Score: {cmi}')
            statistic = cmi
            threshold = _compute_threshold(lmbda_1, lmbda_2, n_candidate_cols)
            logger.debug(f'Calculated Threshold: {threshold}')
            if statistic >= threshold:
                logger.debug(f'Passed, keeping feature {candidate_src}')
            else:
                # ballet.validation.main._prune_existing_features will log
                # this at level INFO
                logger.debug(
                    f'Failed, found redundant feature: {candidate_src}')
                del feature_df_map[candidate_feature]
                redundant_features.append(candidate_feature)
        return redundant_features
