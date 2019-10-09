import numpy as np

from ballet.util import asarray2d

LAMBDA_1_ADJUSTMENT = 64
LAMBDA_2_ADJUSTMENT = 64


def _concat_datasets(dfs_by_src, n_samples=0, omit=None):
    if omit is None:
        omit = []
    filtered_dfs = [np.array(dfs_by_src[x])
                    for x in dfs_by_src if x not in omit]
    if len(filtered_dfs) == 0:
        return np.zeros((n_samples, 1))
    return asarray2d(np.concatenate(filtered_dfs, axis=1))


def _compute_lmbdas(unnorm_lmbda_1, unnorm_lmbda_2, features_by_src):
    feat_srcs = features_by_src.keys()
    num_features = len(feat_srcs)
    num_feature_cols = 0
    for feat_src in features_by_src:
        num_feature_cols += features_by_src[feat_src].shape[1]
    return (unnorm_lmbda_1 / num_features, unnorm_lmbda_2 / num_feature_cols)


def _compute_threshold(lmbda_1, lmbda_2, n_feature_cols, n_omitted_cols=0):
    return lmbda_1 + lmbda_2 * (n_feature_cols - n_omitted_cols)
