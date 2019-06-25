import numpy as np
import scipy.stats
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors

from ballet.util import asarray2d
from ballet.util.log import logger

NUM_NEIGHBORS = 3  # Used by sklearn NearestNeighbors


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
    n_samples, _ = x.shape
    _, counts = np.unique(x, axis=0, return_counts=True)
    pk = counts * 1.0 / n_samples
    return scipy.stats.entropy(pk)


def estimate_cont_entropy(x, epsilon=None):
    """Estimate the differential entropy of a continuous dataset.

    Based off the Kraskov Estimator [1] and Kozachenko [2] estimators for a
    dataset's differential entropy. If epsilon is not provided, this will be the
    Kozacheko Estimator of the dataset's entropy. If epsilon is provided, this
    is a partial estimation of the Kraskov entropy estimator. The bias is
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

    Returns:
        float: differential entropy of the dataset

    References:

    .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
           information". Phys. Rev. E 69, 2004.
    .. [2] L. F. Kozachenko, N. N. Leonenko, "Sample Estimate of the Entropy
           of a Random Vector:, Probl. Peredachi Inf., 23:2 (1987), 9-16

    """
    x = asarray2d(x)
    n_samples, n_features = x.shape
    if n_samples <= 1:
        return 0
    nn = NearestNeighbors(
        metric='chebyshev',
        n_neighbors=NUM_NEIGHBORS,
        algorithm='kd_tree')
    nn.fit(x)
    if epsilon is None:
        # If epsilon is not provided, revert to the Kozachenko Estimator
        n_neighbors = NUM_NEIGHBORS
        radius = 0
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
    if col.dtype == int:
        return True
    rounding_error = col - col.astype(int)
    if np.allclose(rounding_error, np.zeros(col.size)):
        return True
    uniques = np.unique(col)
    return (uniques.size / col.size) < 0.05


def _get_discrete_columns(x):
    return np.apply_along_axis(_is_column_discrete, 0, x)


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
       H(X) = H(c,d) = \sum_{x \in d} p(x) H(c(x)) + H(d)

    Where c(x) is a dataset that represents the rows of the continuous dataset
    in the same row as a discrete column with value x in the original dataset.

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
    if n_features < 1:
        return 0
    disc_mask = _get_discrete_columns(x)
    cont_mask = ~disc_mask
    # If our dataset is fully discrete/continuous, do something easier
    if np.all(disc_mask):
        return estimate_disc_entropy(x)
    elif np.all(cont_mask):
        return estimate_cont_entropy(x, epsilon)

    # Separate the dataset into discrete and continuous datasets d,c
    disc_features = asarray2d(x[:, disc_mask])
    cont_features = asarray2d(x[:, cont_mask])

    entropy = 0
    uniques, counts = np.unique(disc_features, axis=0, return_counts=True)
    empirical_p = counts / n_samples

    # $\sum_{x \in d} p(x) H(c(x))$
    for i in range(counts.size):
        unique_mask = np.all(disc_features == uniques[i], axis=1)
        selected_cont_samples = cont_features[unique_mask, :]
        if epsilon is None:
            selected_epsilon = None
        else:
            selected_epsilon = epsilon[unique_mask, :]
        conditional_cont_entropy = estimate_cont_entropy(
            selected_cont_samples, selected_epsilon)
        entropy += empirical_p[i] * conditional_cont_entropy

    # H(d)
    entropy += estimate_disc_entropy(disc_features)
    if epsilon is None:
        entropy = max(0, entropy)
    return entropy


def _calculate_epsilon(x):
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
    disc_mask = _get_discrete_columns(x)
    if np.all(disc_mask):
        # if all discrete columns, there's no point getting epsilon
        return 0
    cont_features = x[:, ~disc_mask]
    nn = NearestNeighbors(metric='chebyshev', n_neighbors=NUM_NEIGHBORS)
    nn.fit(cont_features)
    distances, _ = nn.kneighbors()
    epsilon = np.nextafter(distances[:, -1], 0)
    return asarray2d(epsilon)


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

    This calculation is *exact* for entirely discrete datasets and
    *approximate* if there are continuous columns present.

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
    h_xz = estimate_entropy(xz, epsilon)
    h_yz = estimate_entropy(yz, epsilon)
    h_xyz = estimate_entropy(xyz, epsilon)
    h_z = estimate_entropy(z, epsilon)
    logger.debug('H(X,Z): {}'.format(h_xz))
    logger.debug('H(Y,Z): {}'.format(h_yz))
    logger.debug('H(X,Y,Z): {}'.format(h_xyz))
    logger.debug('H(Z): {}'.format(h_z))
    return max(0, h_xz + h_yz - h_xyz - h_z)


def estimate_mutual_information(x, y):
    r"""Estimate the mutual information of two datasets.

    Mutual information is a measure of dependence between
    two datasets and is calculated as:

    .. math::
       I(x;y) = H(x) + H(y) - H(x,y)

    Where H(x) is the Shannon entropy of x. For continuous datasets,
    adapts the Kraskov Estimator [1] for mutual information. This calculation
    is *exact* for entirely discrete datasets and *approximate* if there are
    continuous columns present.

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
    h_x = estimate_entropy(x, epsilon)
    h_y = estimate_entropy(y, epsilon)
    h_xy = estimate_entropy(xy, epsilon)
    return max(0, h_x + h_y - h_xy)
