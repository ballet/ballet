import random

from ballet.util import asarray2d
from ballet.util.log import logger
from ballet.util.testing import seeded
from ballet.validation.base import FeatureAcceptanceMixin, FeatureAccepter
from ballet.validation.common import RandomFeaturePerformanceEvaluator
from ballet.validation.entropy import (
    estimate_conditional_information, estimate_entropy)
from ballet.validation.gfssf import (
    LAMBDA_1_ADJUSTMENT, LAMBDA_2_ADJUSTMENT, _compute_lmbdas,
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


class GFSSFAccepter(FeatureAccepter):
    """A feature acceptance evaluator that uses a modified version of
    GFSSF[1] - specifically, lines 1 - 8 of agGFSSF where we do not
    remove accepted but redundant features on line 8.

    Attributes:
        X_df (array-like): The dataset to build features off of.
        y (array-like): A single-column dataset representing the target
            feature.
        features (array-like): an array of ballet Features that have
            already been accepted.
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

    def __init__(self, *args, lmbda_1=0., lmbda_2=0.):
        super().__init__(*args)
        self.y = asarray2d(self.y)
        if lmbda_1 <= 0:
            lmbda_1 = estimate_entropy(self.y) / LAMBDA_1_ADJUSTMENT
        if lmbda_2 <= 0:
            lmbda_2 = estimate_entropy(self.y) / LAMBDA_2_ADJUSTMENT
        self.lmbda_1 = lmbda_1
        self.lmbda_2 = lmbda_2

    def __str__(self):
        return '{str}: lmbda_1={lmbda_1}, lmbda_2={lmbda_2}'.format(
            str=super().__str__(),
            lmbda_1=self.lmbda_1,
            lmbda_2=self.lmbda_2)

    def judge(self):
        logger.info('Judging Feature using {}'.format(self))
        feature_dfs_by_src = {}
        for feature in [self.candidate_feature] + self.features:
            feature_df = (feature
                          .as_feature_engineering_pipeline()
                          .fit_transform(self.X_df, self.y))
            feature_dfs_by_src[feature.source] = feature_df

        candidate_source = self.candidate_feature.source
        candidate_df = feature_dfs_by_src[candidate_source]
        n_samples, n_candidate_cols = feature_df.shape

        lmbda_1, lmbda_2 = _compute_lmbdas(
            self.lmbda_1, self.lmbda_2, feature_dfs_by_src)

        logger.info(
            'Candidate Feature Shape: {}'.format(candidate_df.shape))
        omit_in_test = [''] + [f.source for f in self.features]
        for omit in omit_in_test:
            logger.debug(
                'Testing with omitted feature: {}'.format(
                    omit or 'None'))
            z = _concat_datasets(
                feature_dfs_by_src, n_samples, [
                    candidate_source, omit])
            logger.debug('Calculating CMI of candidate feature:')
            cmi = estimate_conditional_information(candidate_df, self.y, z)
            logger.debug(
                'Conditional Mutual Information Score: {}'.format(cmi))
            cmi_omit = 0
            n_omit_cols = 0
            if omit:
                omit_df = feature_dfs_by_src[omit]
                _, n_omit_cols = omit_df.shape
                logger.debug(
                    'Calculating CMI of ommitted feature:'
                )
                cmi_omit = estimate_conditional_information(
                    omit_df, self.y, z)
                logger.debug('Omitted CMI Score: {}'.format(cmi_omit))
                logger.debug('Omitted Feature Shape: {}'.format(omit_df.shape))
            statistic = cmi - cmi_omit
            threshold = _compute_threshold(
                lmbda_1, lmbda_2, n_candidate_cols, n_omit_cols)
            logger.debug('Calculated Threshold: {}'.format(threshold))
            if statistic >= threshold:
                logger.debug(
                    'Succeeded while omitting feature: {}'.format(
                        omit or 'None'))
                return True
        return False
