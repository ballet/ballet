import random

import funcy as fy

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
        n_samples, n_candidate_cols = candidate_df.shape

        lmbda_1, lmbda_2 = _compute_lmbdas(
            self.lmbda_1, self.lmbda_2, feature_dfs_by_src)

        logger.debug(
            f'Recomputed lambda_1={lmbda_1:0.4f}, lambda_2={lmbda_2:0.4f}')

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
                _, n_omit_cols = omit_df.shape

                # Calculate CMI of omitted feature
                cmi_omit = estimate_conditional_information(
                    omit_df, self.y, z)
            else:
                cmi_omit = 0
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
                    n_samples=n_samples,
                    candidate_name=candidate_source,
                    candidate_cols=n_candidate_cols,
                    candidate_cmi=cmi,
                    omitted_name=omitted_feature,
                    omitted_cols=n_omit_cols,
                    omitted_cmi=cmi_omit,
                    statistic=statistic,
                    threshold=threshold,
                    delta=delta,
                )
                info.append(iteration_info)
                logger.debug(
                    f'Completed iteration {i}/{n_omit}: {iteration_info}')

        info_closest = max(info, key=lambda x: x.delta)
        cmi_closest = info_closest.candidate_cmi
        omitted_cmi_closest = info_closest.omitted_cmi
        statistic_closest = info_closest.statistic
        threshold_closest = info_closest.threshold
        logger.info(
            f'Rejected feature: marginal conditional mutual information was not greater than threshold ({cmi_closest:0.4f}-{omitted_cmi_closest:0.4f}={statistic_closest:0.4f} vs {threshold_closest:0.4f} in best case).')  # noqa

        return False
