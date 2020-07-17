from sklearn.pipeline import FeatureUnion

from ballet.eng.base import GroupedFunctionTransformer

__all__ = ['SingleLagger', 'make_multi_lagger']


class SingleLagger(GroupedFunctionTransformer):
    """Transformer that applies a lag operator to each group

    Args:
        lag (int): lag to apply
        groupby_kwargs (dict): keyword arguments to pd.DataFrame.groupby
    """

    def __init__(self, lag, groupby_kwargs=None):
        super().__init__(lambda x: x.shift(lag), groupby_kwargs=groupby_kwargs)


def make_multi_lagger(lags, groupby_kwargs=None):
    """Return a union of transformers that apply different lags

    Args:
        lags (Collection[int]): collection of lags to apply
        groupby_kwargs (dict): keyword arguments to pd.DataFrame.groupby
    """
    laggers = [
        SingleLagger(lag, groupby_kwargs=groupby_kwargs)
        for lag in lags
    ]
    feature_union = FeatureUnion([
        (repr(lagger), lagger) for lagger in laggers
    ])
    return feature_union
