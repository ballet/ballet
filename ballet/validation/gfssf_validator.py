import math

import numpy as np
import pandas as pd
from scipy.special import digamma, gamma
from sklearn.neighbors import NearestNeighbors

from ballet.feature import make_mapper
from ballet.validation.base import FeatureAcceptanceEvaluator

NUM_NEIGHBORS = 3  # Used in the sklearn mutual information function


def _calculate_disc_entropy(X):
    # An exact calculation of the dataset entropy, using empirical probability
    # H = sum(p_i * log2(p_i))
    n_samples, _ = X.shape
    _, counts = np.unique(X, axis=0, return_counts=True)
    empirical_p = counts * 1.0 / n_samples
    log_p = np.log(empirical_p)
    return -np.sum(np.multiply(empirical_p, log_p))


def _estimate_cont_entropy(X, epsilon=None):
    # Based off the Kraskov Estimator for Shannon Entropy
    # https://journals.aps.org/pre/pdf/10.1103/PhysRevE.69.066138
    # Implementation based off summary here:
    # https://pdfs.semanticscholar.org/b3f6/fb5755bf1fdc0d4e97e3805399d32d433611.pdf
    # Calculating eq. 22 minus the last term (cancels out)
    n_samples, n_features = X.shape
    if n_samples <= 1:
        return 0
    if epsilon is None:
        epsilon = _calculate_epsilon(X)
    nn = NearestNeighbors(
        metric='chebyshev',
        n_neighbors=NUM_NEIGHBORS,
        algorithm='kd_tree')
    nn.fit(X)
    ind = nn.radius_neighbors(radius=epsilon, return_distance=False)
    nx = np.array([i.size for i in ind])
    log_vd = n_features * math.log(math.pi) - \
        math.log(gamma(n_features / 2.0 + 1))
    return log_vd - np.mean(digamma(nx + 1)) + digamma(n_samples)


def _is_column_discrete(col):
    # Stand-in method to figure out if column is discrete
    # Still researching good ways to do this...
    uniques = np.unique(col)
    return (uniques.size / col.size) < 0.05


def _get_discrete_columns(X):
    return np.apply_along_axis(_is_column_discrete, 0, X)


def _estimate_entropy(X, epsilon=None):
    n_samples, n_features = X.shape
    if n_features < 1:
        return 0
    if epsilon is None:
        epsilon = _calculate_epsilon(X)
    disc_mask = _get_discrete_columns(X)
    cont_mask = ~disc_mask
    # If our dataset is fully disc/cont, do something easier
    if np.all(disc_mask):
        return _calculate_disc_entropy(X)
    elif np.all(cont_mask):
        return _estimate_cont_entropy(X, epsilon.ravel())

    disc_features = X[:, disc_mask]
    cont_features = X[:, cont_mask]

    entropy = 0
    uniques, counts = np.unique(disc_features, axis=0, return_counts=True)
    empirical_p = counts / n_samples
    log_p = np.log(empirical_p)
    for i in range(counts.size):
        unique_mask = disc_features == uniques[i]
        selected_cont_samples = np.reshape(
            cont_features[unique_mask.ravel(), :], (counts[i], -1))
        selected_epsilon = epsilon[unique_mask]
        conditional_cont_entropy = _estimate_cont_entropy(
            selected_cont_samples, selected_epsilon)
        entropy += empirical_p[i] * (conditional_cont_entropy - log_p[i])
    return entropy


def _calculate_epsilon(X):
    disc_mask = _get_discrete_columns(X)
    if np.all(disc_mask):
        # if all discrete columns, there's no point getting epsilon
        return 0
    cont_features = X[:, ~disc_mask]
    nn = NearestNeighbors(metric='chebyshev', n_neighbors=NUM_NEIGHBORS)
    nn.fit(cont_features)
    distances, _ = nn.kneighbors()
    epsilon = np.nextafter(distances[:, -1], 0)
    return np.reshape(epsilon, (epsilon.size, -1))


def _estimate_conditional_information(x, y, z):
    """
    Estimates I(x;y|z) = H(x,z) + H(y,z) - H(x,y,z) - H(z)
    Is **exact** for entirely discrete columns
    and **approximate** if there are continuous columns present
    """
    xz = np.concatenate((x, z), axis=1)
    yz = np.concatenate((y, z), axis=1)
    xyz = np.concatenate((xz, y), axis=1)
    epsilon = _calculate_epsilon(xyz)
    h_xz = _estimate_entropy(xz, epsilon)
    h_yz = _estimate_entropy(yz, epsilon)
    h_xyz = _estimate_entropy(xyz, epsilon)
    h_z = _estimate_entropy(z, epsilon)
    return max(0, h_xz + h_yz - h_xyz - h_z)


def _concat_datasets(dfs_by_src, omit=None):
    filtered_dfs = [np.array(dfs_by_src[x])
                    for x in dfs_by_src if x is not omit]
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
        _, n_feature_clms = feature_df.shape
        n_feature_clms_arr = n_feature_clms
        n_feature_grps_arr = 1
        feature_dfs_by_src = {}
        for accepted_feature in self.features:
            accepted_df = make_mapper(
                [feature]).fit_transform(
                self.X_df, self.y)
            n_feature_clms_arr += accepted_df.shape[1]
            n_feature_grps_arr += 1
            feature_dfs_by_src[accepted_feature.src] = accepted_df

        lmbda_1 = self.lmbda_1 / n_feature_grps_arr
        lmbda_2 = self.lmbda_2 / n_feature_clms_arr
        omit_in_test = [''] + [f.src for f in self.features]
        for omit in omit_in_test:
            z = _concat_datasets(feature_dfs_by_src, omit)
            cmi = _estimate_conditional_information(feature_df, self.y, z)

            cmi_omit = 0
            n_clms_omit = 0
            if omit is not '':
                omit_df = feature_dfs_by_src[omit]
                cmi_omit = _estimate_conditional_information(
                    omit_df, self.y, z)
                _, n_clms_omit = omit_df.shape
            statistic = cmi - cmi_omit
            threshold = lmbda_1 + \
                lmbda_2 * (n_feature_clms - n_clms_omit)
            if statistic >= threshold:
                return True
        return False
