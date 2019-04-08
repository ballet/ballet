from ballet.util import asarray2d
from ballet.util.log import logger
from ballet.validation.base import FeatureAcceptanceEvaluator
from ballet.validation.entropy import (
    estimate_conditional_information, estimate_entropy)
from ballet.validation.gfssf import (
    LAMBDA_1_ADJUSTMENT, LAMBDA_2_ADJUSTMENT, _compute_lmbdas,
    _compute_threshold, _concat_datasets)


class NoOpAcceptanceEvaluator(FeatureAcceptanceEvaluator):

    def judge(self, feature):
        return True


class GFSSFAcceptanceEvaluator(FeatureAcceptanceEvaluator):
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

    def __init__(self, X_df, y, features, lmbda_1=0., lmbda_2=0.):
        super().__init__(X_df, y, features)
        self.y = asarray2d(y)
        if (lmbda_1 <= 0):
            lmbda_1 = estimate_entropy(self.y) / LAMBDA_1_ADJUSTMENT
        if (lmbda_2 <= 0):
            lmbda_2 = estimate_entropy(self.y) / LAMBDA_2_ADJUSTMENT
        self.lmbda_1 = lmbda_1
        self.lmbda_2 = lmbda_2

    def judge(self, candidate_feature):
        feature_dfs_by_src = {}
        for feature in [candidate_feature] + self.features:
            feature_df = feature.as_dataframe_mapper().fit_transform(
                self.X_df, self.y)
            feature_dfs_by_src[feature.source] = feature_df

        candidate_source = candidate_feature.source
        candidate_df = feature_dfs_by_src[candidate_source]
        n_samples, n_candidate_cols = feature_df.shape

        lmbda_1, lmbda_2 = _compute_lmbdas(
            self.lmbda_1, self.lmbda_2, feature_dfs_by_src)

        logger.info(
            'Judging Feature using GFSSF: lambda_1={l1}, lambda_2={l2}'.format(
                l1=lmbda_1, l2=lmbda_2))
        omit_in_test = [''] + [f.source for f in self.features]
        for omit in omit_in_test:
            logger.debug(
                'Testing with omitted feature: {}'.format(
                    omit or 'None'))
            z = _concat_datasets(
                feature_dfs_by_src, n_samples, [
                    candidate_source, omit])
            cmi = estimate_conditional_information(candidate_df, self.y, z)
            logger.debug(
                'Conditional Mutual Information Score: {}'.format(cmi))
            cmi_omit = 0
            n_omit_cols = 0
            if omit:
                omit_df = feature_dfs_by_src[omit]
                _, n_omit_cols = omit_df.shape
                cmi_omit = estimate_conditional_information(
                    omit_df, self.y, z)
                logger.debug('Omitted CMI Score: {}'.format(cmi_omit))
            statistic = cmi - cmi_omit
            threshold = _compute_threshold(
                lmbda_1, lmbda_2, n_candidate_cols, n_omit_cols)
            logger.debug('Calculated Threshold: {}'.format(threshold))
            if statistic >= threshold:
                logger.debug(
                    'Succeeded while ommitting feature: {}'.format(
                        omit or 'None'))
                return True
        return False
