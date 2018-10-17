import json

import numpy as np
from sklearn.linear_model import LinearRegression

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


def get_p_value(X, y, xi):
    N = np.size(y, 0)
    ones = np.ones((N, 1))
    if X is None:
        X0 = ones
    else:
        X0 = np.hstack((ones, X))

    X1 = np.hstack((X, asarray2d(xi)))

    model0 = LinearRegression(fit_intercept=False)
    model0.fit(X0, y)
    ypred0 = model0.predict(X0)
    error0 = np.sum(np.square(y - ypred0))

    model1 = LinearRegression(fit_intercept=False)
    model1.fit(X1, y)
    ypred1 = model1.predict(X1)
    error1 = np.sum(np.square(y - ypred1))

    p = np.exp((error1 - error0) / (2 * error0 / N))

    assert p > 0
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
