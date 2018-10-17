import json

import numpy as np
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
        p = get_p_value(X, y, xi)
        logger.debug('Got p value {p!r}'.format(p=p))
        return p < self.ai


def compute_log_likelihood(X, y, hasconst=True):
    return OLS(y, X, hasconst=hasconst).fit().llf


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

    # T ~ chi2Ck)
    T = -2 * (logL0 - logL1)
    p = 1 - chi2.cdf(T, k)

    assert p >= 0

    return p


def update_ai(a, i, accepted):
    "Given a_i, i, accepted_i, produces a_{i+1}"
    w = a * (2 * (i - 1))
    w = w - a + da * accepted
    i += 1
    return w / (2 * i)


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
