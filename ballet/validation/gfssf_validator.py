import numpy as np
import pandas as pd

from ballet.feature import make_mapper
from ballet.validation.base import FeatureAcceptanceEvaluator


def _calculate_entropy_discrete(X):
    pass

def _estimate_entropy_continuous(X):
    pass

def _estimate_entropy(X):
    pass

def _estimate_conditional_information(x, y, z):
    """
    Estimates I(x;y|z) = H(x,z) + H(y,z) - H(x,y,z) - H(z)
    Is **exact** for entirely discrete columns
    and **approximate** if there are continuous columns present
    """
    pass

def _concat_datasets(dfs_by_src, omit=None):
    pass

class GFSSFAcceptanceEvaluator(FeatureAcceptanceEvaluator):
    def __init__(self, X_df, y, features, lmbda_1=0, lmbda_2=0):
        super().__init__(self, X_df, y, features)
        if (lmbda_1 <= 0):
            lmbda_1 = _estimate_entropy(y) / 4.0
        if (lmbda_2 <= 0):
            lmbda_2 = _estimate_entropy(y) / 4.0
        self.lmbda_1 = lmbda_1
        self.lmbda_2 = lmbda_2
    
    def judge(self, feature):
        feature_df = make_mapper([feature]).fit_transform(self.X_df, self.y)
        feature_dfs_by_src = {}
        for accepted_feature in self.features:
            accepted_df = make_mapper([feature]).fit_transform(self.X_df, self.y)
            feature_dfs_by_src[accepted_feature.src] = accepted_df
        omit_in_test = [''] + map(lambda f: f.src, self.features)
        for omit in omit_in_test:
            z = _concat_datasets(feature_dfs_by_src, omit)
            cmi = _estimate_conditional_information(feature_df, self.y, z)


