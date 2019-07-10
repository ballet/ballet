import numpy as np
import scipy.stats
from funcy import decorator
from scipy.special import digamma, gamma
from sklearn.neighbors import NearestNeighbors

from ballet.util import asarray2d
from ballet.util.log import logger

__all__ = (
    'estimate_conditional_information',
    'estimate_entropy',
    'estimate_mutual_information',
)

N_NEIGHBORS = 3   # hyperparameter k from Kraskov estimator
NEIGHBORS_ALGORITHM = 'kd_tree'
NEIGHBORS_METRIC = 'chebyshev'
DISC_COL_UNIQUE_VAL_THRESH = 0.05


@decorator
def nonnegative(call, name=None):
    result = call()
    if result < 0:
        msg = '{result} should be non-negative.'.format(
            result=name or 'Result')
        logger.warn(msg)
        result = 0
    return result


# Helpers

def _make_neighbors(**kwargs):
    return NearestNeighbors(algorithm=NEIGHBORS_ALGORITHM,
                            metric=NEIGHBORS_METRIC,
                            **kwargs)


def _compute_empirical_probability(x):
    n, _ = x.shape
    _, counts = np.unique(x, axis=0, return_counts=True)
    return counts * 1.0 / n


def _compute_volume_unit_ball(d, metric=NEIGHBORS_METRIC):
    """Compute volume of a d-dimensional unit ball in R^d with given metric"""
    if metric == 'chebyshev':
        return 1
    elif metric == 'euclidean':
        return np.power(np.pi, d / 2) / gamma(1 + d / 2) / 2**d
    else:
        msg = 'metric {metric} not supported'.format(metric=metric)
        raise ValueError(msg)


def _is_column_disc(col):
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


def _is_column_cont(col):
    return not _is_column_disc(col)


def _get_disc_columns(x):
    return np.apply_along_axis(_is_column_disc, 0, x)


# Computing epsilon

def _compute_epsilon(x, cont_method):
    # TODO
    if cont_method == 'kraskov':
        _compute_epsilon_kraskov(x)
    elif cont_method == 'kl':
        _compute_epsilon_kl(x)
    else:
        raise ValueError


def _compute_epsilon_kraskov(x, n_neighbors=N_NEIGHBORS):
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
    nn = _make_neighbors(n_neighbors=n_neighbors)
    nn.fit(cont_features)
    dist, _ = nn.kneighbors()
    dist = dist[:, -1]  # distances to kth nearest points
    epsilon = np.nextafter(dist, 0)
    return asarray2d(epsilon)


def _compute_epsilon_kl(x, n_neighbors=N_NEIGHBORS):
    """Calculate epsilon """
    n, d = x.shape
    k = n_neighbors
    nn = _make_neighbors(n_neighbors=k)
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
        raise ValueError("All samples were the same, can't calculate epsilon")

    epsilon = 2 * distances

    return epsilon


def _compute_n_points_within_epsilon(x, epsilon):
    """Compute the number of points within a distance of epsilon"""
    radius = epsilon.ravel()
    nn = _make_neighbors(radius=radius)
    nn.fit(x)
    ind = nn.radius_neighbors(return_distance=False)
    nx = np.array([i.size for i in ind])
    return nx


def _recover_k_from_epsilon(x, epsilon):
    nx = _compute_n_points_within_epsilon(x, epsilon)
    return min(nx)


# Entropy estimation

def _estimate_disc_entropy(x):
    r"""Estimate the Shannon entropy of a discrete dataset.

    The Shannon entropy of a discrete random variable :math:`Z` with support
    :math:`\mathbb{Z}` and density :math:`P_Z` is given as

    .. math::
        H(Z) = -\sum_{z \in \mathbb{Z}} P_Z(z) \log(P_Z(z))

    Here, since we do not know :math:`P_Z`, we estimate :math:`\hat{P}_Z`, the
    empirical probability, calculated as the frequency in the dataset x.

    If x's columns logically represent continuous features, it is better to use
    the `_estimate_cont_entropy` function. If you are unsure of which to use,
    `estimate_entropy` can take datasets of mixed discrete and continuous
    functions.

    Args:
        x (array-like): Dataset with shape (n_samples, n_features) or
            (n_samples, )

    Returns:
        float: the dataset entropy.
    """
    x = asarray2d(x)
    pk = _compute_empirical_probability(x)
    return scipy.stats.entropy(pk)


def _estimate_cont_entropy(x, cont_method, epsilon,
                           n_neighbors=N_NEIGHBORS):
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
    the _estimate_disc_entropy function. If you are unsure of which to use,
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
    if cont_method == 'kl':
        return _estimate_cont_entropy_kl(x, epsilon, n_neighbors)
    elif cont_method == 'kraskov':
        return _estimate_cont_entropy_kraskov(x, epsilon, n_neighbors)
    else:
        raise ValueError(
            'Invalid method: {cont_method}'
            .format(cont_method=cont_method))


def _estimate_cont_entropy_kl(x, epsilon, n_neighbors=N_NEIGHBORS):
    """Kraskov et al, Eq 20"""
    # TODO(mjs) - figure out how to provide epsilon and k
    x = asarray2d(x)
    k = n_neighbors
    n, d = x.shape
    # eps_i is twice the distance from x_i to its kth neighbor
    eps = _compute_epsilon_kl(x, k)
    c_d = _compute_volume_unit_ball(d, NEIGHBORS_METRIC)
    return -digamma(k) + digamma(n) + np.log(c_d) + d * np.mean(np.log(eps))


def _estimate_cont_entropy_kraskov(x, epsilon):
    """Kraskov et al, Eq 22"""
    x = asarray2d(x)
    n, d = x.shape
    nx = _compute_n_points_within_epsilon(x, epsilon)
    c_d = _compute_volume_unit_ball(d)
    return -np.mean(digamma(nx + 1)) + digamma(n) + np.log(c_d) \
        + d * np.mean(np.log(epsilon))


def _estimate_entropy(x, cont_method, epsilon):
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
    n, d = x.shape

    # not enough data
    if n <= 1 or d == 0:
        return 0

    disc_mask = _get_disc_columns(x)
    cont_mask = ~disc_mask

    # if all columns are disc, use one estimator
    if np.all(disc_mask):
        return _estimate_disc_entropy(x)

    # if all columns are cont, use one estimator
    if np.all(cont_mask):
        return _estimate_cont_entropy(x, cont_method, epsilon)

    # Separate the dataset into discrete and continuous datasets d,c
    d = asarray2d(x[:, disc_mask])
    c = asarray2d(x[:, cont_mask])

    # H(c|d)
    H_c_d = _estimate_conditional_entropy(c, d, cont_method, epsilon)

    # H(d)
    H_d = _estimate_disc_entropy(d)

    return H_d + H_c_d


def _estimate_conditional_entropy(c, d, cont_method, epsilon):
    """Estimate H(c|d) where c is continuous and d is discrete"""
    # H(c|d) = \sum_{i} p(d_i) H(c|d=d_i)
    # where we i ranges over unique values of d and c_d_i is the
    # rows of c where d == d_i.
    pk = _compute_empirical_probability(d)
    uniques, _ = np.unique(d, axis=0, return_counts=True)
    H_c_d = 0.0
    for i, d_i in enumerate(uniques):
        mask = np.all(d == d_i, axis=1)
        c_di = c[mask, :]
        eps_di = epsilon[mask, :]
        H_c_di = _estimate_cont_entropy(c_di, cont_method, eps_di)
        H_c_d += pk[i] * H_c_di
    return H_c_d


# Public API

@nonnegative(name='Entropy')
def estimate_entropy(x):
    x = asarray2d(x)
    cont_method = 'kl'
    epsilon = _compute_epsilon(x, cont_method)
    return _estimate_entropy(x, cont_method, epsilon)


@nonnegative(name='Conditional mutual information')
def estimate_conditional_information(x, y, z):
    r"""Estimate the conditional mutual information of x and y given z

    Conditional mutual information is the mutual information of two datasets,
    given a third:

    .. math::
       I(x;y|z) = H(x,z) + H(y,z) - H(x,y,z) - H(z)

    Where :math:`H(X)` is the Shannon entropy of dataset :math:`X`. For
    continuous datasets, adapts the Kraskov Estimator [1] for mutual
    information.

    Eq 8 from [1] holds because the epsilon terms cancel out.
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

    cont_method = 'kraskov'
    epsilon = _compute_epsilon(xyz, cont_method)

    H_xz = _estimate_entropy(xz, cont_method, epsilon)
    H_yz = _estimate_entropy(yz, cont_method, epsilon)
    H_xyz = _estimate_entropy(xyz, cont_method, epsilon)
    H_z = _estimate_entropy(z, cont_method, epsilon)

    logger.debug('H(X,Z): {}'.format(H_xz))
    logger.debug('H(Y,Z): {}'.format(H_yz))
    logger.debug('H(X,Y,Z): {}'.format(H_xyz))
    logger.debug('H(Z): {}'.format(H_z))

    return H_xz + H_yz - H_xyz - H_z


@nonnegative(name='Mutual information')
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
    cont_method = 'kraskov'
    epsilon = _compute_epsilon(xy, cont_method)
    H_x = _estimate_entropy(x, cont_method, epsilon)
    H_y = _estimate_entropy(y, cont_method, epsilon)
    H_xy = _estimate_entropy(xy, cont_method, epsilon)
    return H_x + H_y - H_xy
