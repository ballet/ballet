from typing import List, NamedTuple

import numpy as np
import pandas as pd
from funcy import iterable
from sklearn_pandas import DataFrameMapper

import ballet.feature
from ballet.eng import BaseTransformer
from ballet.eng.misc import NullTransformer
from ballet.feature import Feature


class FeatureEngineeringPipeline(DataFrameMapper):
    """Feature engineering pipeline

    Args:
        features (Union[Feature, List[Feature]]): feature or list of features
    """

    def __init__(self, features):
        if not features:
            features = ballet.feature.Feature(input=[],
                                              transformer=NullTransformer())

        if not iterable(features):
            features = (features, )

        super().__init__(
            [t.as_input_transformer_tuple() for t in features],
            input_df=True)


class BuildResult(NamedTuple):
    X_df: pd.DataFrame
    features: List[Feature]
    mapper_X: FeatureEngineeringPipeline
    X: np.array
    y_df: pd.DataFrame
    encoder_y: BaseTransformer
    y: np.array
