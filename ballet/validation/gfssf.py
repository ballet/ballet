from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ballet.feature import Feature
from ballet.util import asarray2d
from ballet.validation.base import FeaturePerformanceEvaluator
from ballet.validation.entropy import estimate_entropy

LAMBDA_1_ADJUSTMENT = 64
LAMBDA_2_ADJUSTMENT = 64


def _concat_datasets(
    feature_df_map: Dict[Feature, pd.DataFrame],
    n_samples: int = 0,
    omit: Optional[List[Feature]] = None
) -> np.ndarray:
    if omit is None:
        omit = []

    filtered_dfs = [
        np.array(feature_df_map[feature])
        for feature in feature_df_map
        if feature not in omit
    ]

    if not filtered_dfs:
        return np.zeros((n_samples, 1))

    return asarray2d(np.concatenate(filtered_dfs, axis=1))


def _compute_lmbdas(
    unnorm_lmbda_1: float,
    unnorm_lmbda_2: float,
    features_by_src: Dict[str, np.ndarray],
) -> Tuple[float, float]:
    num_features = len(features_by_src)
    num_feature_cols = sum(
        features_by_src[feat_src].shape[1]
        for feat_src in features_by_src
    )
    # if there are no features, then don't adjust the lambdas
    # num_feature_cols = max(1, num_feature_cols)
    lmbda_1 = unnorm_lmbda_1 / num_features
    lmbda_2 = unnorm_lmbda_2 / num_feature_cols
    return lmbda_1, lmbda_2


def _compute_threshold(
    lmbda_1: float,
    lmbda_2: float,
    n_feature_cols: int,
    n_omitted_cols: int = 0
) -> float:
    return lmbda_1 + lmbda_2 * (n_feature_cols - n_omitted_cols)


@dataclass
class GFSSFIterationInfo:
    i: int
    n_samples: int
    candidate_feature: Feature
    candidate_cols: int
    candidate_cmi: float
    omitted_feature: Feature
    omitted_cols: int
    omitted_cmi: float
    statistic: float
    threshold: float
    delta: float

    def __str__(self):
        def format(v):
            if isinstance(v, (float, np.float_)):
                return f'{v:.4e}'
            elif isinstance(v, Feature):
                return v.source or '<live object>'
            return str(v)
        return ', '.join(
            f'{k}={format(v)}'
            for k, v in self.__dict__.items()
        )


class GFSSFPerformanceEvaluator(FeaturePerformanceEvaluator):
    """A feature performance evaluator that uses a modified version of GFSSF[1]

    Attributes:
        lmbda_1: GFSSF parameter used to calculate the information
            threshold. Default is a function of the entropy of y.
        lmbda_2: GFSSF parameter used to calculate the information
            threshold. Default is a function of the entropy of y.
        lambda_1_adjustment: Adjustment to estimated entropy used to
            calculate lmbda_1.
        lambda_2_adjustment: Adjustment to estimated entropy used to
            calculate lmbda_2.

    References:
        [1] H. Li, X. Wu, Z. Li and W. Ding, "Group Feature Selection
            with Streaming Features," 2013 IEEE 13th International
            Conference on Data Mining, Dallas, TX, 2013, pp. 1109-1114.
            doi: 10.1109/ICDM.2013.137
    """

    def __init__(
        self,
        *args,
        lmbda_1: float = 0.0,
        lmbda_2: float = 0.0,
        lambda_1_adjustment: float = LAMBDA_1_ADJUSTMENT,
        lambda_2_adjustment: float = LAMBDA_2_ADJUSTMENT
    ):
        super().__init__(*args)
        self.y = asarray2d(self.y)
        if lmbda_1 <= 0:
            lmbda_1 = estimate_entropy(self.y) / lambda_1_adjustment
        if lmbda_2 <= 0:
            lmbda_2 = estimate_entropy(self.y) / lambda_2_adjustment
        self.lmbda_1 = lmbda_1
        self.lmbda_2 = lmbda_2

    def __str__(self):
        cls = super().__str__()
        return \
            f'{cls}: lmbda_1={self.lmbda_1:0.4f}, lmbda_2={self.lmbda_2:0.4f}'

    def _get_feature_df_map(self):
        all_features = [*self.features, self.candidate_feature]

        def as_features(feature):
            return (
                feature
                .as_feature_engineering_pipeline()
                .fit_transform(self.X_df, y=self.y_df))

        # map feature defintion "id" -> feature values
        feature_df_map = {
            feature: as_features(feature)
            for feature in all_features
        }

        return feature_df_map
