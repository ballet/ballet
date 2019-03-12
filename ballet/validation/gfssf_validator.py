import math

import numpy as np
import pandas as pd
from scipy.special import digamma, gamma
from sklearn.neighbors import NearestNeighbors

from ballet.feature import make_mapper
from ballet.util import asarray2d
from ballet.util.log import logger
from ballet.validation.base import FeatureAcceptanceEvaluator

NUM_NEIGHBORS = 3  # Used in the sklearn mutual information function


def _calculate_disc_entropy(X):
    # An exact calculation of the dataset entropy, using empirical probability
    # H = sum(p_i * log2(p_i))
    n_samples, _ = X.shape
    _, counts = np.unique(X, axis=0, return_counts=True)
    empirical_p = counts * 1.0 / n_samples
    log_p = np.log(empirical_p)
    entropy = -np.sum(np.multiply(empirical_p, log_p))
    return entropy


def _estimate_cont_entropy(X, epsilon=None):
    """
    Based off the Kraskov Estimator for Shannon Entropy
    https://journals.aps.org/pre/pdf/10.1103/PhysRevE.69.066138
    Implementation based off summary here:
    https://pdfs.semanticscholar.org/b3f6/fb5755bf1fdc0d4e97e3805399d32d433611.pdf
    Calculating eq. 22 minus the last term (cancels out)
    """
    n_samples, n_features = X.shape
    if n_samples <= 1:
        return 0
    nn = NearestNeighbors(
        metric='chebyshev',
        n_neighbors=NUM_NEIGHBORS,
        algorithm='kd_tree')
    nn.fit(X)
    if epsilon is None:
        # If epsilon is not provided, revert to the Kozachenko Estimator
        n_neighbors = NUM_NEIGHBORS
        radius = None
        # While we have non-zero radii, calculate for a larger k
        # Potentially expensive
        while not np.all(radius) and n_neighbors < n_samples:
            distances, _ = nn.kneighbors(
                n_neighbors=n_neighbors, return_distance=True)
            radius = distances[:, -1]
            n_neighbors += 1
        if n_neighbors == n_samples:
            # This case only happens if all samples are the same
            # e.g. this isn't a continuous sample...
            raise ValueError('Should not have discrete column to estimate')
        return -digamma(n_neighbors) + digamma(n_samples) + \
            n_features * np.mean(np.log(2 * radius))
    else:
        ind = nn.radius_neighbors(
            radius=epsilon.ravel(),
            return_distance=False)
        nx = np.array([i.size for i in ind])
        return - np.mean(digamma(nx + 1)) + digamma(n_samples)


def _is_column_discrete(col):
    # Stand-in method to figure out if column is discrete
    # Still researching good ways to do this...
    uniques = np.unique(col)
    return (uniques.size / col.size) < 0.05


def _get_discrete_columns(X):
    return np.apply_along_axis(_is_column_discrete, 0, X)


def _estimate_entropy(X, epsilon=None):
    """
    Estimates the entropy of a dataset.
    When epsilon is provided, we instead calculate
    a partial estimation based on the Kraskov Estimator
    When epsilon is NOT provided, we calculate the
    Kozachenko Estimator's full estimation.
    """
    n_samples, n_features = X.shape
    if n_features < 1:
        return 0
    disc_mask = _get_discrete_columns(X)
    cont_mask = ~disc_mask
    # If our dataset is fully disc/cont, do something easier
    if np.all(disc_mask):
        return _calculate_disc_entropy(X)
    elif np.all(cont_mask):
        return _estimate_cont_entropy(X, epsilon)

    disc_features = asarray2d(X[:, disc_mask])
    cont_features = asarray2d(X[:, cont_mask])

    entropy = 0
    uniques, counts = np.unique(disc_features, axis=0, return_counts=True)
    empirical_p = counts / n_samples
    log_p = np.log(empirical_p)
    for i in range(counts.size):
        unique_mask = disc_features == uniques[i]
        selected_cont_samples = cont_features[unique_mask.ravel(), :]
        if epsilon is not None:
            selected_epsilon = epsilon[unique_mask]
        else:
            selected_epsilon = None
        conditional_cont_entropy = _estimate_cont_entropy(
            selected_cont_samples, selected_epsilon)
        entropy += empirical_p[i] * (conditional_cont_entropy - log_p[i])
    if epsilon is None:
        entropy = max(0, entropy)
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
    return asarray2d(epsilon)


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


def _concat_datasets(dfs_by_src, n_samples, omit=None):
    filtered_dfs = [np.array(dfs_by_src[x])
                    for x in dfs_by_src if x is not omit]
    if len(filtered_dfs) == 0:
        return np.zeros((n_samples, 1))
    return asarray2d(np.concatenate(filtered_dfs, axis=1))


class GFSSFAcceptanceEvaluator(FeatureAcceptanceEvaluator):
    def __init__(self, X_df, y, features, lmbda_1=0, lmbda_2=0):
        super().__init__(X_df, y, features)
        self.y = asarray2d(y)
        if (lmbda_1 <= 0):
            lmbda_1 = _estimate_entropy(self.y) / 32
        if (lmbda_2 <= 0):
            lmbda_2 = _estimate_entropy(self.y) / 32
        self.lmbda_1 = lmbda_1
        self.lmbda_2 = lmbda_2

    def judge(self, feature):
        feature_df = make_mapper([feature]).fit_transform(self.X_df, self.y)
        n_samples, n_feature_clms = feature_df.shape
        n_feature_clms_arr = n_feature_clms
        n_feature_grps_arr = 1
        feature_dfs_by_src = {}
        for accepted_feature in self.features:
            accepted_df = make_mapper(
                [feature]).fit_transform(
                self.X_df, self.y)
            n_feature_clms_arr += accepted_df.shape[1]
            n_feature_grps_arr += 1
            feature_dfs_by_src[accepted_feature.source] = accepted_df

        lmbda_1 = self.lmbda_1 / n_feature_grps_arr
        lmbda_2 = self.lmbda_2 / n_feature_clms_arr
        logger.info(
            'Judging Feature using GFSSF: lambda_1={l1}, lambda_2={l2}'.format(
                l1=lmbda_1, l2=lmbda_2))
        omit_in_test = [''] + [f.source for f in self.features]
        for omit in omit_in_test:
            logger.debug(
                'Testing with omitted feature: {}'.format(
                    omit or 'None'))
            z = _concat_datasets(feature_dfs_by_src, n_samples, omit)
            cmi = _estimate_conditional_information(feature_df, self.y, z)
            logger.debug(
                'Conditional Mutual Information Score: {}'.format(cmi))
            cmi_omit = 0
            n_clms_omit = 0
            if omit is not '':
                omit_df = feature_dfs_by_src[omit]
                cmi_omit = _estimate_conditional_information(
                    omit_df, self.y, z)
                _, n_clms_omit = omit_df.shape
                logger.debug('Omitted CMI Score: {}'.format(cmi_omit))
            statistic = cmi - cmi_omit
            threshold = lmbda_1 + \
                lmbda_2 * (n_feature_clms - n_clms_omit)
            logger.debug('Calculated Threshold: {}'.format(threshold))
            if statistic >= threshold:
                logger.debug(
                    'Succeeded while ommitting feature: {}'.format(
                        omit or 'None'))
                return True
        return False
