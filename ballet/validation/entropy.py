import numpy as np
from scipy.special import digamma, gamma
from sklearn.neighbors import NearestNeighbors

from ballet.util import asarray2d

NUM_NEIGHBORS = 3  # Used in the sklearn mutual information function


def calculate_disc_entropy(X):
    # An exact calculation of the dataset entropy, using empirical probability
    # H = sum(p_i * log2(p_i))
    n_samples, _ = X.shape
    _, counts = np.unique(X, axis=0, return_counts=True)
    empirical_p = counts * 1.0 / n_samples
    log_p = np.log(empirical_p)
    entropy = -np.sum(np.multiply(empirical_p, log_p))
    return entropy


def estimate_cont_entropy(X, epsilon=None):
    """
    Based off the Kraskov Estimator for Shannon Entropy
    Implementation based off summary here:
    https://journals.aps.org/pre/pdf/10.1103/PhysRevE.69.066138
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


def estimate_entropy(X, epsilon=None):
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
    # If our dataset is fully discrete/continuous, do something easier
    if np.all(disc_mask):
        return calculate_disc_entropy(X)
    elif np.all(cont_mask):
        return estimate_cont_entropy(X, epsilon)

    disc_features = asarray2d(X[:, disc_mask])
    cont_features = asarray2d(X[:, cont_mask])

    entropy = 0
    uniques, counts = np.unique(disc_features, axis=0, return_counts=True)
    empirical_p = counts / n_samples
    for i in range(counts.size):
        unique_mask = disc_features == uniques[i]
        selected_cont_samples = cont_features[unique_mask.ravel(), :]
        if epsilon is not None:
            selected_epsilon = epsilon[unique_mask]
        else:
            selected_epsilon = None
        conditional_cont_entropy = estimate_cont_entropy(
            selected_cont_samples, selected_epsilon)
        entropy += empirical_p[i] * conditional_cont_entropy
    entropy += calculate_disc_entropy(disc_features)
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


def estimate_conditional_information(x, y, z):
    """
    Estimates I(x;y|z) = H(x,z) + H(y,z) - H(x,y,z) - H(z)
    Is **exact** for entirely discrete columns
    and **approximate** if there are continuous columns present
    """
    xz = np.concatenate((x, z), axis=1)
    yz = np.concatenate((y, z), axis=1)
    xyz = np.concatenate((xz, y), axis=1)
    epsilon = _calculate_epsilon(xyz)
    h_xz = estimate_entropy(xz, epsilon)
    h_yz = estimate_entropy(yz, epsilon)
    h_xyz = estimate_entropy(xyz, epsilon)
    h_z = estimate_entropy(z, epsilon)
    return max(0, h_xz + h_yz - h_xyz - h_z)


def estimate_mutual_information(x, y):
    xy = np.concatenate((x, y), axis=1)
    epsilon = _calculate_epsilon(xy)
    h_x = estimate_entropy(x, epsilon)
    h_y = estimate_entropy(y, epsilon)
    h_xy = estimate_entropy(xy, epsilon)
    return max(0, h_x + h_y - h_xy)
