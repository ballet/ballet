import numpy as np
import pandas as pd

from ballet.feature import make_mapper
from ballet.validation.base import FeatureAcceptanceEvaluator
from scipy.sparse import issparse


def _calculate_entropy_discrete(X):
    uniques = np.unique(X, axis=0)
    
    pass

def _estimate_entropy_continuous(X):
    pass

def _determine_discrete_columns(X):
    pass

def _estimate_entropy(X):
    n_samples, n_features = X.shape
    disc_mask = _determine_discrete_columns(X)
    cont_mask = ~disc_mask
    if np.all(disc_mask):
        return _calculate_entropy_discrete(X)
    elif np.all(cont_mask):
        return _estimate_entropy_continuous(X)

    disc_features = X[:, disc_mask]
    cont_features = X[:, cont_mask]

    conditional_entropy = 0
    
    return conditional_entropy + _calculate_entropy_discrete(disc_features)


def _estimate_conditional_information(x, y, z):
    """
    Estimates I(x;y|z) = H(x,z) + H(y,z) - H(x,y,z) - H(z)
    Is **exact** for entirely discrete columns
    and **approximate** if there are continuous columns present
    """
    xz = np.concatenate((x,z), axis=1)
    yz = np.concatenate((y,z), axis=1)
    xyz = np.concatenate((xz, y), axis=1)
    h_xz = _estimate_entropy(xz)
    h_yz = _estimate_entropy(yz)
    h_xyz = _estimate_entropy(xyz)
    h_z = _estimate_entropy(z)
    return max(0, h_xz + h_yz - h_xyz - h_z)

def _concat_datasets(dfs_by_src, omit=None):
    filtered_srcs = filter(lambda x: x is not omit, dfs_by_src.keys())
    filtered_dfs = map(lambda x: dfs_by_src[x], filtered_srcs)
    return np.concatenate(filtered_dfs, axis=1)

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
        _, n_features = feature_df.shape
        feature_dfs_by_src = {}
        for accepted_feature in self.features:
            accepted_df = make_mapper([feature]).fit_transform(self.X_df, self.y)
            feature_dfs_by_src[accepted_feature.src] = accepted_df
        omit_in_test = [''] + map(lambda f: f.src, self.features)
        for omit in omit_in_test:
            z = _concat_datasets(feature_dfs_by_src, omit)
            cmi = _estimate_conditional_information(feature_df, self.y, z)

            cmi_omit = 0
            n_features_omit = 0
            if omit is not '':
                omit_df = feature_dfs_by_src[omit]
                cmi_omit = _estimate_conditional_information(omit_df, self.y, z)
                _, n_features_omit = omit_df.shape
            statistic = cmi - cmi_omit
            threshold = self.lmbda_1 + self.lmbda_2(n_features - n_features_omit)
            if statistic >= threshold:
                return True
        return False


