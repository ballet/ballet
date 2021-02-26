from typing import Callable, Collection, NamedTuple, Tuple, cast

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
from ballet.util.typing import OneOrMore


class FeatureEngineeringPipeline(DataFrameMapper):
    """Feature engineering pipeline

    Args:
        features: feature or list of features
    """

    def __init__(self, features: OneOrMore['ballet.feature.Feature']):
        if not features:
            _features = [
                ballet.feature.Feature(input=[],
                                       transformer=NullTransformer())
            ]
        elif not iterable(features):
            features = cast(ballet.feature.Feature, features)
            _features = [features, ]
        else:
            features = cast(Collection[ballet.feature.Feature], features)
            _features = list(features)

        self._ballet_features = _features

        super().__init__(
            [t.as_input_transformer_tuple() for t in _features],
            input_df=True)

    @property
    def ballet_features(self) -> Collection['ballet.feature.Feature']:
        return self._ballet_features


class EngineerFeaturesResult(NamedTuple):
    X_df: pd.DataFrame
    features: Collection['ballet.feature.Feature']
    pipeline: FeatureEngineeringPipeline
    X: np.ndarray
    y_df: pd.DataFrame
    encoder: BaseTransformer
    y: np.ndarray


def make_engineer_features(
    pipeline: FeatureEngineeringPipeline,
    encoder: BaseTransformer,
    load_data: Callable[..., Tuple[pd.DataFrame, pd.DataFrame]],
) -> Callable[[pd.DataFrame, pd.DataFrame], EngineerFeaturesResult]:
    features = pipeline.ballet_features

    @stacklog(logger.info, 'Building features and target')
    def engineer_features(
        X_df: pd.DataFrame = None, y_df: pd.DataFrame = None
    ) -> EngineerFeaturesResult:
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
        X = pipeline.fit_transform(X_df, y=y_df)
        y = encoder.fit_transform(y_df)

        return EngineerFeaturesResult(
            X_df=X_df, features=features, pipeline=pipeline, X=X,
            y_df=y_df, encoder=encoder, y=y)

    return engineer_features
