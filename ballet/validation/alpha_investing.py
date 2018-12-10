import json

import numpy as np
from funcy import decorator
from numpy.linalg.linalg import LinAlgError
from scipy.stats import chi2
from statsmodels.api import OLS

from ballet.compat import pathlib
from ballet.feature import make_mapper
from ballet.util import asarray2d
from ballet.util.log import logger
from ballet.validation.base import FeatureAcceptanceEvaluator

w0 = 0.5
da = 0.5


class AlphaInvestingAcceptanceEvaluator(FeatureAcceptanceEvaluator):

    def __init__(self, X_df, y, features, ai):
        super().__init__(X_df, y, features)
        assert ai > 0, 'ai was less than 0: {0}'.format(ai)
        self.ai = ai

    def judge(self, feature):
        logger.info(
            'Judging feature using alpha investing, alpha={ai}'
            .format(ai=self.ai))
        mapper_X = make_mapper(self.features)
        X = mapper_X.fit_transform(self.X_df)
        y = self.y
        mapper_xi = make_mapper(feature)
        xi = mapper_xi.fit_transform(self.X_df)
        X = remove_candidate_feature(X, xi)
        p = get_p_value(X, y, xi)
        logger.debug('Got p value {p!r}'.format(p=p))
        return bool(p < self.ai)


@decorator
def replace_invalid(call, pred, value):
    x = call()
    return x if not pred(x) else value


@replace_invalid(np.isnan, -np.inf)
def compute_log_likelihood(X, y, hasconst=True):
    try:
        return OLS(y, X, hasconst=hasconst).fit().llf
    except LinAlgError:
        logger.debug('Error computing log likelihood; returning -inf')
        return -np.inf


def get_p_value(X, y, xi):
    xi = asarray2d(xi)
    N = np.size(y, 0)
    k = np.size(xi, 1)

    X0 = np.ones((N, 1))
    if X is not None:
        X0 = np.hstack((X0, X))
    X1 = np.hstack((X0, xi))

    logL0 = compute_log_likelihood(X0, y)
    logL1 = compute_log_likelihood(X1, y)

    assert np.isfinite(logL0), f'No reason for logL0 to be {logL0}'

    # T ~ chi2(k)
    T = -2 * (logL0 - logL1)
    p = 1 - chi2.cdf(T, k)

    assert p >= 0, f'p was {p}, L0 was {logL0}, L1 was {logL1}, T was {T}'

    return p

def remove_candidate_feature(X, xi):
    k = np.size(xi, 1)
    n = np.size(X, 1)
    
    for i in range(n - k + 1):
        if np.allclose(X[:, i:(i + k)], xi, rtol= .001):
            return np.concatenate((X[:, :i], X[:, i + k:]), axis=1)
    return X

def update_ai(ai, i, accepted):
    "Given a_i, i, accepted_i, produces a_{i+1}"
    wi = ai * 2 * i
    wi1 = wi - ai + da * accepted
    return wi1 / (2 * (i+1))


def update_wi(w, i, accepted):
    """Given w_i, i, accepted_i, produces w_{i+1}"""
    a = w / (2 * i)
    w = w - a + da * accepted
    return w


def compute_parameters(outcomes, w0=w0, da=da):
    """Compute ai from a list of acceptance outcomes

    Args:
        outcomes (List[str]): list of either "accepted" or "rejected"
        w0 (float): initial wealth
        da (float): additional investment
    """
    ais = []
    wis = []

    w = w0
    wis.append(w)
    i = 1
    for outcome in outcomes:
        a = w / (2 * i)
        ais.append(a)
        w = w - a + da * (outcome == 'accepted')
        wis.append(w)
        i += 1

    # compute a_{i+1} for next iteration
    a = w / (2 * i)
    ais.append(a)

    return ais, wis


def compute_ai(outcomes, w0=w0, da=da):
    ais, wis = compute_parameters(outcomes, w0=w0, da=da)
    return ais[-1]


def get_alpha_file_path(project):
    return pathlib.Path(project.path).joinpath('.alpha.json')


def load_alpha(project):
    alpha_file = get_alpha_file_path(project)
    if not alpha_file.exists():
        with alpha_file.open('w') as f:
            content = {'a': w0 / 2, 'i': 1}
            json.dump(content, f)

    with alpha_file.open('r') as f:
        content = json.load(f)

    return content['a'], content['i']


def save_alpha(project, ai, i, accepted):
    alpha_file = get_alpha_file_path(project)
    content = {
        'a': update_ai(ai, i, accepted),
        'i': i + 1,
    }

    with alpha_file.open('w') as f:
        json.dump(content, f)
