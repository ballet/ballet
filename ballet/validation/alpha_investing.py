import numpy as np
from sklearn.linear_model import LinearRegression

from ballet.feature import make_mapper
from ballet.validation.base import FeatureAcceptanceEvaluator
from ballet.util import asarray2d


class AlphaInvestingAcceptanceEvaluator(FeatureAcceptanceEvaluator):

    w0 = 0.5
    da = 0.5

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
