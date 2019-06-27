import numpy as np
import scipy.stats
from scipy.special import digamma, gamma
from sklearn.neighbors import NearestNeighbors

from ballet.util import asarray2d
from ballet.util.log import logger

N_NEIGHBORS = 3   # hyperparameter k from Kraskov estimator
DISC_COL_UNIQUE_VAL_THRESH = 0.05


def compute_empirical_probability(x):
    n_samples, _ = x.shape
    _, counts = np.unique(x, axis=0, return_counts=True)
    return counts * 1.0 / n_samples


def estimate_disc_entropy(x):
    r"""Estimate the Shannon entropy of a discrete dataset.

    The Shannon entropy of a discrete random variable :math:`Z` with support
    :math:`\mathbb{Z}` and density :math:`P_Z` is given as

    .. math::
        H(Z) = -\sum_{z \in \mathbb{Z}} P_Z(z) \log(P_Z(z))

    Here, since we do not know :math:`P_Z`, we estimate :math:`\hat{P}_Z`, the
    empirical probability, calculated as the frequency in the dataset x.

    If x's columns logically represent continuous features, it is better to use
    the `estimate_cont_entropy` function. If you are unsure of which to use,
    `estimate_entropy` can take datasets of mixed discrete and continuous
    functions.

    Args:
        x (array-like): Dataset with shape (n_samples, n_features) or
            (n_samples, )

    Returns:
        float: the dataset entropy.
    """
    x = asarray2d(x)
    pk = compute_empirical_probability(x)
    return scipy.stats.entropy(pk)


def _compute_volume_unit_ball(d, metric):
    if metric == 'chebyshev':
        return 1
    elif metric == 'euclidean':
        return np.power(np.pi, d / 2) / gamma(1 + d / 2) / 2**d
    else:
        raise NotImplementedError(
            'metric {metric} not supported'.format(metric=metric))


def _estimate_cont_entropy_kozachenko_leonenko(x,
                                               n_neighbors=N_NEIGHBORS,
                                               metric='chebyshev'):
    x = asarray2d(x)
    k = n_neighbors
    n, d = x.shape
    if n <= 1:
        return 0

    # compute eps: eps_i is twice the distance from x_i to its kth neighbor
    eps = _calculate_epsilon_2(x, k, metric)
    c_d = _compute_volume_unit_ball(d, metric)

    # Kraskov et al, Eq 20
    H_hat = -digamma(k) + digamma(n) + np.log(c_d) + d * np.mean(np.log(eps))

    if H_hat < 0:
        logger.warn('Entropy should be non-negative')
        H_hat = 0

    return H_hat


def estimate_cont_entropy(x, epsilon=None, n_neighbors=N_NEIGHBORS):
    """Estimate the differential entropy of a continuous dataset.

    Based off the Kraskov Estimator [1] and Kozachenko [2] estimators for a
    dataset's differential entropy. If epsilon is not provided, this will be
    the Kozacheko Estimator of the dataset's entropy. If epsilon is provided,
    this is a partial estimation of the Kraskov entropy estimator. The bias is
    cancelled out when computing mutual information.

    The function relies on nonparametric methods based on entropy estimation
    from k-nearest neighbors distances as proposed in [1] and augmented in [2]
    for mutual information estimation.

    If X's columns logically represent discrete features, it is better to use
    the estimate_disc_entropy function. If you are unsure of which to use,
    estimate_entropy can take datasets of mixed discrete and continuous
    functions.

    Args:
        x (array-like): Dataset with shape (n_samples, n_features) or
            (n_samples, )
        epsilon (array-like): An array with shape (n_samples, 1) that is
            the epsilon used in Kraskov Estimator. Represents the Chebyshev
            distance from an element to its k-th nearest neighbor in the full
            dataset.
        n_neighbors (int): number of neighbors to use in Kraskov estimator (
            hyperparameter k)

    Returns:
        float: differential entropy of the dataset

    References:

    .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
           information". Phys. Rev. E 69, 2004.
    .. [2] L. F. Kozachenko, N. N. Leonenko, "Sample Estimate of the Entropy
           of a Random Vector:, Probl. Peredachi Inf., 23:2 (1987), 9-16

    """
    x = asarray2d(x)
    k = n_neighbors
    n, d = x.shape
    if n <= 1:
        return 0
    nn = NearestNeighbors(
        metric='chebyshev',
        n_neighbors=k,
        algorithm='kd_tree')
    nn.fit(x)
    if epsilon is None:
        # If epsilon is not provided, revert to the Kozachenko Estimator
        radius = 0
        # While we have non-zero radii, calculate for a larger k
        # Potentially expensive
        while not np.all(radius) and k < n:
            distances, _ = nn.kneighbors(
                n_neighbors=k, return_distance=True)
            radius = distances[:, -1]
            k += 1
        if k == n:
            # This case only happens if all samples are the same
            # e.g. this isn't a continuous sample...
            raise ValueError('Should not have discrete column to estimate')
        return -digamma(k) + digamma(n) + d * np.mean(np.log(2 * radius))
    else:
        ind = nn.radius_neighbors(
            radius=epsilon.ravel(),
            return_distance=False)
        nx = np.array([i.size for i in ind])
        return - np.mean(digamma(nx + 1)) + digamma(n)


def is_column_disc(col):
    # Heuristic to decide if column is discrete

    # Integer columns are discrete
    if col.dtype is int:
        return True

    # Real-valued columns that are close to integer values are discrete
    rounding_error = col - col.astype(int)
    if np.allclose(rounding_error, np.zeros(col.size)):
        return True

    # Columns with a small fraction of distinct values are discrete
    uniques = np.unique(col)
    if (uniques.size / col.size) < DISC_COL_UNIQUE_VAL_THRESH:
        return True

    return False


def is_column_cont(col):
    return not is_column_disc(col)


def _get_disc_columns(x):
    return np.apply_along_axis(is_column_disc, 0, x)


def estimate_entropy(x, epsilon=None):
    r"""Estimate dataset entropy.

    This function can take datasets of mixed discrete and continuous features,
    and uses a set of heuristics to determine which functions to apply to each.
    If the dataset is fully discrete, an exact calculation is done. If this is
    not the case and epsilon is not provided, this will be the Kozacheko
    Estimator of the dataset's entropy. If epsilon is provided, this is a
    partial estimation of the Kraskov entropy estimator. The bias is cancelled
    out when computing mutual information.

    Because this function is a subroutine in a mutual information estimator,
    we employ the Kozachenko Estimator[1] for continuous features when this
    function is _not_ used for mutual information and an adaptation of the
    Kraskov Estimator[2] when it is.

    Let x be made of continuous features c and discrete features d.
    To deal with both continuous and discrete features, We use the
    following reworking of entropy:

    .. math::
       :nowrap:

       \begin{align}
       H(x) &= H(c,d) \\
            &= H(d) + H(c | d) \\
            &= \sum_{x \in d} p(x) H(c(x)) + H(d),
       \end{align}

    where :math:`c(x)` is a dataset that represents the rows of the continuous
    dataset in the same row as a discrete column with value x in the original
    dataset.

    Args:
        x (array-like): Dataset with shape (n_samples, n_features) or
            (n_samples, )
        epsilon (array-like): An array with shape (n_samples, 1) that is
            the epsilon used in Kraskov Estimator. Represents the chebyshev
            distance from an element to its k-th nearest neighbor in the full
            dataset.

    Returns:
        float: Dataset entropy of X.

    References:

    .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
           information". Phys. Rev. E 69, 2004.
    .. [2] L. F. Kozachenko, N. N. Leonenko, "Sample Estimate of the Entropy
           of a Random Vector:, Probl. Peredachi Inf., 23:2 (1987), 9-16.
    """
    x = asarray2d(x)
    n_samples, n_features = x.shape

    # presumed empty array
    if n_features == 0:
        return 0

    disc_mask = _get_disc_columns(x)
    cont_mask = ~disc_mask

    # if all columns are disc, use specific estimator
    if np.all(disc_mask):
        return estimate_disc_entropy(x)

    # if all columns are cont, use specific estimator
    if np.all(cont_mask):
        return estimate_cont_entropy(x, epsilon)

    # Separate the dataset into discrete and continuous datasets d,c
    d = asarray2d(x[:, disc_mask])
    c = asarray2d(x[:, cont_mask])

    pk = compute_empirical_probability(d)
    d_uniques, d_counts = np.unique(d, axis=0, return_counts=True)

    # H(c|d) = \sum_{x \in d} p(x) H(c(x))
    H_c_d = 0
    for i in range(d_counts.size):
        unique_mask = np.all(d == d_uniques[i], axis=1)
        selected_cont_samples = c[unique_mask, :]
        if epsilon is None:
            selected_epsilon = None
        else:
            selected_epsilon = epsilon[unique_mask, :]
        conditional_cont_entropy = estimate_cont_entropy(
            selected_cont_samples, selected_epsilon)
        H_c_d += pk[i] * conditional_cont_entropy

    # H(d)
    H_d = estimate_disc_entropy(d)

    H = H_d + H_c_d

    if epsilon is None:
        if H < 0:
            logger.warn('Entropy should be non-negative')
            H = 0

    return H


def _calculate_epsilon(x, n_neighbors=N_NEIGHBORS, metric='chebyshev'):
    """Calculate epsilon from Kraskov Estimator

    Represents the Chebyshev distance of each dataset element to its
    k-th nearest neighbor.

    Args:
        X (array-like): An array with shape (n_samples, n_features)

    Returns:
        array-like: An array with shape (n_samples, 1) representing
            epsilon as described above.

    References:

    .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
           information". Phys. Rev. E 69, 2004.

    """
    disc_mask = _get_disc_columns(x)
    if np.all(disc_mask):
        # if all discrete columns, there's no point getting epsilon
        return 0
    cont_features = x[:, ~disc_mask]
    nn = NearestNeighbors(metric=metric, n_neighbors=n_neighbors)
    nn.fit(cont_features)
    distances, _ = nn.kneighbors()
    epsilon = np.nextafter(distances[:, -1], 0)
    return asarray2d(epsilon)


def _calculate_epsilon_2(x, n_neighbors, metric):
    n, d = x.shape
    k = n_neighbors
    nn = NearestNeighbors(
        metric=metric,
        n_neighbors=k,
        algorithm='kd_tree')
    nn.fit(x)
    distances = 0
    # While we have non-zero radii, calculate for a larger k
    # Potentially expensive
    while not np.all(distances) and k < n:
        distances, _ = nn.kneighbors(n_neighbors=k, return_distance=True)
        distances = distances[:, -1]
        k += 1

    if k == n:
        # This case only happens if all samples are the same
        # e.g. this isn't a continuous sample...
        raise ValueError("All samples were the same, can't calculate espilon")

    epsilon = 2 * distances

    return epsilon


def estimate_conditional_information(x, y, z):
    r"""Estimate the conditional mutual information of x and y given z

    Conditional mutual information is the mutual information of two datasets,
    given a third:

    .. math::
       I(x;y|z) = H(x,z) + H(y,z) - H(x,y,z) - H(z)

    Where :math:`H(X)` is the Shannon entropy of dataset :math:`X`. For
    continuous datasets, adapts the Kraskov Estimator [1] for mutual
    information.

    Equation 8 from [1] holds because the epsilon terms cancel out.
    Let :math:`d_x`, represent the dimensionality of the continuous portion of
    x. Then, we see that:

    .. math::
       :nowrap:

       \begin{align}
       d_{xz} + d_{yz} - d_{xyz} - d_z
           &= (d_x + d_z) + (d_y + d_z) - (d_x + d_y + d_z) - d_z \\
           &= 0
       \end{align}

    Args:
        x (array-like): An array with shape (n_samples, n_features_x)
        y (array-like): An array with shape (n_samples, n_features_y)
        z (array-like): An array with shape (n_samples, n_features_z).
            This is the dataset being conditioned on.

    Returns:
        float: conditional mutual information of x and y given z.

    References:

    .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
           information". Phys. Rev. E 69, 2004.
    """
    xz = np.concatenate((x, z), axis=1)
    yz = np.concatenate((y, z), axis=1)
    xyz = np.concatenate((xz, y), axis=1)

    epsilon = _calculate_epsilon(xyz)

    H_xz = estimate_entropy(xz, epsilon)
    H_yz = estimate_entropy(yz, epsilon)
    H_xyz = estimate_entropy(xyz, epsilon)
    H_z = estimate_entropy(z, epsilon)

    logger.debug('H(X,Z): {}'.format(H_xz))
    logger.debug('H(Y,Z): {}'.format(H_yz))
    logger.debug('H(X,Y,Z): {}'.format(H_xyz))
    logger.debug('H(Z): {}'.format(H_z))

    MI = H_xz + H_yz - H_xyz - H_z

    if MI < 0:
        logger.warn('Mutual information should be non-negative')
        MI = 0

    return MI


def estimate_mutual_information(x, y):
    r"""Estimate the mutual information of two datasets.

    Mutual information is a measure of dependence between
    two datasets and is calculated as:

    .. math::
       I(x;y) = H(x) + H(y) - H(x,y)

    Where H(x) is the Shannon entropy of x. For continuous datasets,
    adapts the Kraskov Estimator [1] for mutual information.

    Args:
        x (array-like): An array with shape (n_samples, n_features_x)
        y (array-like): An array with shape (n_samples, n_features_y)

    Returns:
        float: mutual information of x and y

    References:

    .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
      information". Phys. Rev. E 69, 2004.
    """
    xy = np.concatenate((x, y), axis=1)
    epsilon = _calculate_epsilon(xy)
    H_x = estimate_entropy(x, epsilon)
    H_y = estimate_entropy(y, epsilon)
    H_xy = estimate_entropy(xy, epsilon)

    MI = H_x + H_y - H_xy

    if MI < 0:
        logger.warn('Mutual information should be non-negative')
        MI = 0

    return MI
