import math

import numpy as np
import pandas as pd

from ballet.feature import make_mapper
from ballet.validation.base import FeatureAcceptanceEvaluator

from scipy.special import gamma, digamma
from sklearn.neighbors import NearestNeighbors

NUM_NEIGHBORS=3 # Used in the sklearn mutual information function

def _calculate_disc_entropy(X):
    n_samples, _ = X.shape
    _, counts = np.unique(X, axis=0, return_counts=True)
    empirical_p = counts * 1.0 / n_samples
    log_p = np.log2(empirical_p)
    return np.sum(np.multiply(empirical_p, log_p))

def _estimate_cont_entropy(X):
    n_samples, n_features = X.size
    nn = NearestNeighbors(metric='chebyshev',n_neighbors=NUM_NEIGHBORS)
    nn.fit(X)
    r = np.array(nn.kneighbors(X, return_distance=True)).flatten()
    sum_log_r = np.sum(np.log2(r)) * n_features / n_samples
    log_vd = n_features * math.log2(math.pi / gamma(n_features / 2.0 + 1)) / 2.0
    return max(0, sum_log_r + log_vd - digamma(NUM_NEIGHBORS) + digamma(n_samples))

def _is_column_discrete(col):
    # Stand-in method to figure out if column is discrete
    # Still researching good ways to do this...
    uniques = np.unique(col)
    return (uniques.size * 1.0 / col.size) < 0.05 

def _estimate_entropy(X):
    n_samples, _ = X.shape
    disc_mask = np.apply_along_axis(_is_column_discrete, 0, X)
    cont_mask = ~disc_mask
    if np.all(disc_mask):
        return _calculate_disc_entropy(X)
    elif np.all(cont_mask):
        return _estimate_cont_entropy(X)

    disc_features = X[:, disc_mask]
    cont_features = X[:, cont_mask]

    entropy = 0
    uniques, counts = np.unique(X, axis=0, return_counts=True)
    empirical_p = counts * 1.0 / n_samples
    log_p = np.log2(empirical_p)
    for i in range(counts.size):
        unique_mask = disc_features == uniques[i]
        selected_cont_samples = cont_features[unique_mask, :]
        conditional_cont_entropy = _estimate_cont_entropy(selected_cont_samples)
        entropy += empirical_p[i] * (conditional_cont_entropy + log_p[i])
    return log_p


def _estimate_conditional_information(x, y, z):
    """
    Estimates I(x;y|z) = H(x,z) + H(y,z) - H(x,y,z) - H(z)
    Is **exact** for entirely discrete columns
    and **approximate** if there are continuous columns present
    """
    xz = np.concatenate((x,z), axis=1)
    yz = np.concatenate((y,z), axis=1)
    xyz = np.concatenate((xz, y), axis=1)
    h_xz = _estimate_entropy(xz)
    h_yz = _estimate_entropy(yz)
    h_xyz = _estimate_entropy(xyz)
    h_z = _estimate_entropy(z)
    return max(0, h_xz + h_yz - h_xyz - h_z)

def _concat_datasets(dfs_by_src, omit=None):
    filtered_srcs = filter(lambda x: x is not omit, dfs_by_src.keys())
    filtered_dfs = map(lambda x: np.array(dfs_by_src[x]), filtered_srcs)
    return np.concatenate(filtered_dfs, axis=1)

class GFSSFAcceptanceEvaluator(FeatureAcceptanceEvaluator):
    def __init__(self, X_df, y, features, lmbda_1=0, lmbda_2=0):
        super().__init__(self, X_df, y, features)
        if (lmbda_1 <= 0):
            lmbda_1 = _estimate_entropy(y) / 4.0
        if (lmbda_2 <= 0):
            lmbda_2 = _estimate_entropy(y) / 4.0
        self.lmbda_1 = lmbda_1
        self.lmbda_2 = lmbda_2
    
    def judge(self, feature):
        feature_df = make_mapper([feature]).fit_transform(self.X_df, self.y)
        _, n_features = feature_df.shape
        feature_dfs_by_src = {}
        for accepted_feature in self.features:
            accepted_df = make_mapper([feature]).fit_transform(self.X_df, self.y)
            feature_dfs_by_src[accepted_feature.src] = accepted_df
        omit_in_test = [''] + map(lambda f: f.src, self.features)
        for omit in omit_in_test:
            z = _concat_datasets(feature_dfs_by_src, omit)
            cmi = _estimate_conditional_information(feature_df, self.y, z)

            cmi_omit = 0
            n_features_omit = 0
            if omit is not '':
                omit_df = feature_dfs_by_src[omit]
                cmi_omit = _estimate_conditional_information(omit_df, self.y, z)
                _, n_features_omit = omit_df.shape
            statistic = cmi - cmi_omit
            threshold = self.lmbda_1 + self.lmbda_2(n_features - n_features_omit)
            if statistic >= threshold:
                return True
        return False


