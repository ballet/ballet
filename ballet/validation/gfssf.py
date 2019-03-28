import numpy as np

from ballet.util import asarray2d

LAMBDA_1_ADJUSTMENT = 32
LAMBDA_2_ADJUSTMENT = 32


def _concat_datasets(dfs_by_src, n_samples, omit=None):
    filtered_dfs = [np.array(dfs_by_src[x])
                    for x in dfs_by_src if x is not omit]
    if len(filtered_dfs) == 0:
        return np.zeros((n_samples, 1))
    return asarray2d(np.concatenate(filtered_dfs, axis=1))


def _compute_lmbdas(unnorm_lmbda_1, unnorm_lmbda_2, feature_df, acc_by_src):
    feat_srcs = acc_by_src.keys()
    num_features = 1 + len(feat_srcs)
    num_feature_cols = feature_df.shape[1]
    for acc_feat_src in feat_srcs:
        num_feature_cols += acc_by_src[acc_feat_src].shape[1]
    return (unnorm_lmbda_1 / num_features, unnorm_lmbda_2 / num_feature_cols)


def _compute_threshold(lmbda_1, lmbda_2, n_feature_cols, n_omitted_cols):
    return lmbda_1 + lmbda_2 * (n_feature_cols - n_omitted_cols)
