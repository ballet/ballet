import random
from typing import List

import numpy as np

from ballet.util import asarray2d
from ballet.util.log import logger
from ballet.util.testing import seeded
from ballet.validation.base import FeatureAcceptanceMixin, FeatureAccepter
from ballet.validation.common import (
    RandomFeaturePerformanceEvaluator, load_spec,)
from ballet.validation.entropy import (
    estimate_conditional_information, estimate_mutual_information,)
from ballet.validation.gfssf import (
    GFSSFIterationInfo, GFSSFPerformanceEvaluator, _compute_lmbdas,
    _compute_threshold, _concat_datasets,)


class NeverAccepter(FeatureAccepter):

    def judge(self):
        logger.info(f'Judging feature using {self}')
        return False


class RandomAccepter(FeatureAcceptanceMixin,
                     RandomFeaturePerformanceEvaluator):

    def judge(self):
        """Accept feature with probability p"""
        logger.info(f'Judging feature using {self}')
        with seeded(self.seed):
            return random.uniform(0, 1) < self.p


class AlwaysAccepter(FeatureAccepter):
    def judge(self):
        logger.info(f'Judging feature using {self}')
        return True


class GFSSFAccepter(FeatureAcceptanceMixin, GFSSFPerformanceEvaluator):

    def judge(self):
        """Judge feature acceptance using GFSSF

        Uses lines 1-8 of agGFSSF where we do not remove accepted but
        redundant features on line 8.
        """

        logger.info(f'Judging feature using {self}')

        feature_df_map = self._get_feature_df_map()

        candidate_df = feature_df_map[self.candidate_feature]
        n_samples, n_candidate_cols = candidate_df.shape

        lmbda_1, lmbda_2 = _compute_lmbdas(
            self.lmbda_1, self.lmbda_2, feature_df_map)

        logger.debug(
            f'Recomputed lambda_1={lmbda_1:0.3e}, lambda_2={lmbda_2:0.3e}')

        info = []

        omit_in_test = [None, *self.features]
        n_omit = len(omit_in_test)
        for i, omitted_feature in enumerate(omit_in_test):

            z = _concat_datasets(
                feature_df_map, n_samples,
                omit=[self.candidate_feature, omitted_feature])

            # Calculate CMI of candidate feature
            cmi = estimate_conditional_information(candidate_df, self.y_val, z)

            if omitted_feature is not None:
                omit_df = feature_df_map[omitted_feature]
                _, n_omit_cols = omit_df.shape

                # Calculate CMI of omitted feature
                cmi_omit = estimate_conditional_information(
                    omit_df, self.y_val, z)
            else:
                cmi_omit = 0
                n_omit_cols = 0

                # want to log to INFO only the case of I(Z|Y;X) where X is the
                # entire feature matrix, i.e. no omitted features.
                logger.info(f'I(feature ; target | existing_features) = {cmi}')

            statistic = cmi - cmi_omit
            threshold = _compute_threshold(
                lmbda_1, lmbda_2, n_candidate_cols, n_omit_cols)
            delta = statistic - threshold

            if delta >= 0:
                omitted_source = getattr(omitted_feature, 'source', 'None')
                logger.debug(
                    f'Succeeded while omitting feature: {omitted_source}')

                return True
            else:
                iteration_info = GFSSFIterationInfo(
                    i=i,
                    n_samples=n_samples,
                    candidate_feature=self.candidate_feature,
                    candidate_cols=n_candidate_cols,
                    candidate_cmi=cmi,
                    omitted_feature=omitted_feature,
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
            f'Rejected feature: best marginal conditional mutual information was not greater than threshold ({cmi_closest:0.3e} - {omitted_cmi_closest:0.3e} = {statistic_closest:0.3e}, vs needed {threshold_closest:0.3e}).')  # noqa

        return False


class VarianceThresholdAccepter(FeatureAccepter):
    """Accept features with variance above a threshold

    Args:
        threshold: variance threshold
    """

    def __init__(self, *args, threshold=0.05):
        super().__init__(*args)
        self.threshold = threshold

    def judge(self):
        logger.info(f'Judging feature using {self}')
        z = (
            self.candidate_feature
            .as_feature_engineering_pipeline()
            .fit(self.X_df, y=self.y_df)
            .transform(self.X_df_val)
        )
        var = np.var(z, axis=0)
        delta = var - self.threshold
        outcome = np.all(delta > 0)
        logger.info(
            f'Feature variance is {var} vs. threshold {self.threshold} '
            f'({delta} above threshold)')
        return outcome

    def __str__(self):
        return f'{super().__str__()}: threshold={self.threshold}'


class MutualInformationAccepter(FeatureAccepter):
    """Accept features with mutual information with the target above a threshold

    Args:
        threshold: mutual information threshold
        handle_nan_targets: one of ``'fail'`` or ``'ignore'``, whether to
            fail validation if NaN-valued targets are discovered or to drop
            those rows in calculation of the mutual information score
    """

    def __init__(self, *args, threshold=0.05, handle_nan_targets='fail'):
        super().__init__(*args)
        self.threshold = threshold
        self.handle_nan_targets = handle_nan_targets

    def judge(self):
        logger.info(f'Judging feature using {self}')
        z = (
            self.candidate_feature
            .as_feature_engineering_pipeline()
            .fit(self.X_df, y=self.y_df)
            .transform(self.X_df_val)
        )
        y = self.y_val
        z, y = asarray2d(z), asarray2d(y)
        z, y = self._handle_nans(z, y)
        if z is None and y is None:
            # nans were found and handle_nan_targets == 'fail'
            return False
        mi = estimate_mutual_information(z, y)
        delta = mi - self.threshold
        outcome = delta > 0
        logger.info(
            f'Mutual information with target I(Z;Y) is {mi} vs. '
            f'threshold {self.threshold} ({delta} above threshold)')
        return outcome

    def _handle_nans(self, z, y):
        nans = np.any(np.isnan(y), 1)  # whether there are any nans in this row
        if np.any(nans):
            if self.handle_nan_targets == 'fail':
                return None, None  # hack
            elif self.handle_nan_targets == 'ignore':
                z = z[~nans, :]
                y = y[~nans, :]
            else:
                raise ValueError(
                    'Invalid value for handle_nan_targets: '
                    f'{self.handle_nan_targets}'
                )

        return z, y

    def __str__(self):
        return f'{super().__str__()}: threshold={self.threshold}'


class CompoundAccepter(FeatureAccepter):
    """A compound accepter that runs a list of individual accepters

    An accepter spec is just a simple serialization of a class and its kwargs::

        name: ballet.validation.feature_acceptance.validator.CompoundAccepter
        params:
          agg: any
          specs:
            - name: ballet.validation.feature_acceptance.validator.VarianceThresholdAccepter
              params:
                threshold: 0.1
            - name: ballet.validation.feature_acceptance.validator.MutualInformationAccepter
              params:
                threshold: 0.1

    Args:
        agg: one of ``'all'`` or ``'any'``; whether to accept if all
            underlying accepters accept or if any accepter accepts.
        specs: list of dicts of accepter specs
    """  # noqa

    def __init__(self, *args, agg='all', specs: List[dict] = []):
        super().__init__(*args)
        self._agg = agg
        self._specs = specs
        if not self._specs:
            raise ValueError('Missing list of accepter specs!')
        self.accepters = []
        for spec in self._specs:
            cls, params = load_spec(spec)
            self.accepters.append(cls(*args, **params))
        if self._agg == 'all':
            self.agg = all
        elif self._agg == 'any':
            self.agg = any
        else:
            raise ValueError(
                f'Unsupported value for parameter agg: {self._agg}')

    def judge(self):
        logger.info(f'Judging feature using {self}')
        outcomes = {
            accepter.__class__.__name__: accepter.judge()
            for accepter in self.accepters
        }
        logger.debug(f'Got outcomes {outcomes!r} from underlying accepters')
        return self.agg(outcomes.values())

    def __str__(self):
        accepter_str = ', '.join(str(accepter) for accepter in self.accepters)
        return f'{super().__str__()}: '\
            f'agg={self._agg}, accepters=({accepter_str})'
