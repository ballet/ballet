import numpy as np
from sklearn.linear_model import LinearRegression

from ballet.feature import make_mapper
from ballet.validation.base import FeatureAcceptanceEvaluator
from ballet.util import asarray2d


w0 = 0.5
da = 0.5



class AlphaInvestingAcceptanceEvaluator(FeatureAcceptanceEvaluator):

    def __init__(self, *args, ai):
        super().__init__(*args)
        self.ai = ai

    def judge(self, feature):
        mapper_X = make_mapper(self.features)
        X = mapper_X.fit_transform(self.X_df)
        y = self.y
        mapper_xi = make_mapper(feature)
        xi = mapper_xi.fit_transform(self.X_df)
        p = get_p_value(X, y, xi)
        return p < self.ai


def get_p_value(X, y, xi):
    N = np.size(y, 0)
    if X is None:
        X0 = np.ones((N, 1))
    else:
        X0 = X

    X1 = np.hstack((X, asarray2d(xi)))

    model0 = LinearRegression()
    model0.fit(X0, y)
    error0 = 1 - model0.score(X0, y)

    model1 = LinearRegression()
    model1.fit(X1, y)
    error1 = 1 - model1.score(X1, y)

    p = np.exp((error1-error0)*N/(2*error0))
    assert p>0
    return p

def update_ai(a, i, accepted):
    "Given a_i, i, accepted_i, produces a_{i+1}"
    w = a*(2*(i-1))
    w = w - a + da * accepted
    i += 1
    return w/(2*i)


def update_wi(w, i, accepted):
    """Given w_i, i, accepted_i, produces w_{i+1}"""
    a = w/(2*i)
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
        a = w/(2*i)
        ais.append(a)
        w = w - a + da * (outcome == 'accepted')
        wis.append(w)
        i += 1

    # compute a_{i+1} for next iteration
    a = w/(2*i)
    ais.append(a)

    return ais, wis


def compute_ai(outcomes, w0=w0, da=da):
    ais, wis = compute_parameters(outcomes, w0=w0, da=da)
    return ais[-1]
