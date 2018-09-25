from sklearn.pipeline import FeatureUnion

from ballet.eng.base import GroupedFunctionTransformer

__all__ = ['SingleLagger', 'make_multi_lagger']


class SingleLagger(GroupedFunctionTransformer):
    def __init__(self, lag, groupby_kwargs=None):
        super().__init__(lambda x: x.shift(lag), groupby_kwargs=groupby_kwargs)


def make_multi_lagger(lags, groupby_kwargs=None):
    laggers = [SingleLagger(l, groupby_kwargs=groupby_kwargs) for l in lags]
    feature_union = FeatureUnion([
        (repr(lagger), lagger) for lagger in laggers
    ])
    return feature_union
