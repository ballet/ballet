from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ballet.util import asarray2d

LAMBDA_1_ADJUSTMENT = 64
LAMBDA_2_ADJUSTMENT = 64


def _concat_datasets(
    dfs_by_src: Dict[str, pd.DataFrame],
    n_samples: int = 0,
    omit: Optional[List[str]] = None
) -> np.ndarray:
    if omit is None:
        omit = []
    filtered_dfs = [np.array(dfs_by_src[x])
                    for x in dfs_by_src if x not in omit]
    if len(filtered_dfs) == 0:
        return np.zeros((n_samples, 1))
    return asarray2d(np.concatenate(filtered_dfs, axis=1))


def _compute_lmbdas(
    unnorm_lmbda_1: float,
    unnorm_lmbda_2: float,
    features_by_src: Dict[str, np.ndarray],
) -> Tuple[float, float]:
    feat_srcs = features_by_src.keys()
    num_features = len(feat_srcs)
    num_feature_cols = 0
    for feat_src in features_by_src:
        num_feature_cols += features_by_src[feat_src].shape[1]
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
