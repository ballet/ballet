from typing import Iterable, NamedTuple, Union

import numpy as np
import pandas as pd
from funcy import iterable
from sklearn_pandas import DataFrameMapper
from stacklog import stacklog

# n.b. cannot import Feature here bc of circular import
import ballet.feature
from ballet.eng import BaseTransformer
from ballet.eng.misc import NullTransformer
from ballet.util.log import logger


class FeatureEngineeringPipeline(DataFrameMapper):
    """Feature engineering pipeline

    Args:
        features: feature or list of features
    """

    def __init__(self, features: Union['ballet.feature.Feature',
                                       Iterable['ballet.feature.Feature']]):
        if not features:
            features = ballet.feature.Feature(input=[],
                                              transformer=NullTransformer())

        if not iterable(features):
            features = (features, )

        self._ballet_features = features

        super().__init__(
            [t.as_input_transformer_tuple() for t in features],
            input_df=True)

    @property
    def ballet_features(self) -> Iterable['ballet.feature.Feature']:
        return self._ballet_features


class BuildResult(NamedTuple):
    X_df: pd.DataFrame
    features: Iterable['ballet.feature.Feature']
    pipeline: FeatureEngineeringPipeline
    X: np.array
    y_df: pd.DataFrame
    encoder: BaseTransformer
    y: np.array


def make_build(features, encoder, load_data):

    @stacklog(logger.info, 'Building features and target')
    def build(
        X_df: pd.DataFrame = None, y_df: pd.DataFrame = None
    ) -> BuildResult:
        """Build features and target

        Args:
            X_df: raw variables
            y_df: raw target

        Returns:
            build result
        """
        if X_df is None or y_df is None:
            _X_df, _y_df = load_data()
        if X_df is None:
            X_df = _X_df
        if y_df is None:
            y_df = _y_df

        pipeline = FeatureEngineeringPipeline(features)
        X = pipeline.fit_transform(X_df)
        y = encoder.fit_transform(y_df)

        return BuildResult(X_df=X_df, features=features, pipeline=pipeline,
                           X=X, y_df=y_df, encoder=encoder, y=y)
