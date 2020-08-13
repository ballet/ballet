import random

from ballet.util.log import logger
from ballet.util.testing import seeded
from ballet.validation.base import FeatureAcceptanceMixin, FeatureAccepter
from ballet.validation.common import RandomFeaturePerformanceEvaluator
from ballet.validation.entropy import estimate_conditional_information
from ballet.validation.gfssf import (
    GFSSFIterationInfo, GFSSFPerformanceEvaluator, _compute_lmbdas,
    _compute_threshold, _concat_datasets)


class NeverAccepter(FeatureAccepter):

    def judge(self):
        logger.info('Judging feature using {}'.format(self))
        return False


class RandomAccepter(FeatureAcceptanceMixin,
                     RandomFeaturePerformanceEvaluator):

    def judge(self):
        """Accept feature with probability p"""
        logger.info('Judging feature using {}'.format(self))
        with seeded(self.seed):
            return random.uniform(0, 1) < self.p


class AlwaysAccepter(FeatureAccepter):
    def judge(self):
        logger.info('Judging feature using {}'.format(self))
        return True


class GFSSFAccepter(FeatureAcceptanceMixin, GFSSFPerformanceEvaluator):

    def judge(self):
        """Judge feature acceptance using GFSSF

        Uses lines 1-8 of agGFSSF where we do not remove accepted but
        redundant features on line 8.
        """

        logger.info(f'Judging feature using {self}')

        feature_dfs_by_src = self._get_feature_dfs_by_src()

        candidate_source = self.candidate_feature.source
        candidate_df = feature_dfs_by_src[candidate_source]
        candidate_shape = candidate_df.shape
        n_samples, n_candidate_cols = candidate_shape

        lmbda_1, lmbda_2 = _compute_lmbdas(
            self.lmbda_1, self.lmbda_2, feature_dfs_by_src)

        logger.info(f'Recomputed lambda_1={lmbda_1}, lambda_2={lmbda_2}')

        info = []

        omit_in_test = [None] + [f.source for f in self.features]
        n_omit = len(omit_in_test)
        for i, omitted_feature in enumerate(omit_in_test):

            z = _concat_datasets(
                feature_dfs_by_src, n_samples,
                [candidate_source, omitted_feature])

            # Calculate CMI of candidate feature
            cmi = estimate_conditional_information(candidate_df, self.y, z)

            if omitted_feature is not None:
                omit_df = feature_dfs_by_src[omitted_feature]
                omitted_shape = omit_df.shape
                _, n_omit_cols = omitted_shape

                # Calculate CMI of omitted feature
                cmi_omit = estimate_conditional_information(
                    omit_df, self.y, z)
            else:
                cmi_omit = 0
                omitted_shape = None
                n_omit_cols = 0

            statistic = cmi - cmi_omit
            threshold = _compute_threshold(
                lmbda_1, lmbda_2, n_candidate_cols, n_omit_cols)
            delta = statistic - threshold

            if delta >= 0:
                logger.debug(
                    f'Succeeded while omitting feature: {omitted_feature!r}')
                return True
            else:
                iteration_info = GFSSFIterationInfo(
                    i=i,
                    candidate_name=candidate_source,
                    candidate_shape=candidate_shape,
                    candidate_cmi=cmi,
                    omitted_name=omitted_feature,
                    omitted_shape=omitted_shape,
                    omitted_cmi=cmi_omit,
                    statistic=statistic,
                    threshold=threshold,
                    delta=delta,
                )
                info.append(iteration_info)
                logger.debug(
                    f'Completed iteration {i}/{n_omit}: {iteration_info}')

        cmi_max = max(info, key=lambda x: x.candidate_cmi)
        delta_max = max(info, key=lambda x: x.delta)
        logger.info(
            f'Rejected feature: cmi_max={cmi_max}, delta_max={delta_max}')

        return False
