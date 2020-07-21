from os import PathLike
from types import ModuleType
from typing import Optional, Union

import pandas as pd
from funcy import cached_property

from ballet.feature import Feature
from ballet.project import FeatureEngineeringProject, Project
from ballet.validation.common import subsample_data_for_validation
from ballet.validation.feature_api import validate_feature_api
from ballet.validation.main import _load_class


class Client:
    """User client for validating features

    Provides a simple interface to validating a given feature.

    Args:
        prj: a way to create a ballet ``Project``, either with an
            already-created object, the project's top-level module, or a path
            within the project. If none of these are provided, will attempt to
            detect the project by ascending from the current working directory.
    """

    def __init__(
        self,
        prj: Union[Project, ModuleType, str, PathLike, None] = None
    ):
        self._prj = prj

    @cached_property
    def project(self) -> Project:
        """Access ballet-specific project info"""
        if self._prj is None:
            return Project.from_cwd()
        else:
            if isinstance(self._prj, Project):
                return self._prj
            elif isinstance(self._prj, ModuleType):
                return Project(self._prj)
            elif isinstance(self._prj, (str, PathLike)):
                return Project.from_path(self._prj)
            else:
                raise ValueError('not supported')

    @property
    def api(self) -> FeatureEngineeringProject:
        """Access feature engineering API of this project"""
        return self.project.api

    def validate_feature_api(
        self,
        feature: Feature,
        X_df: Optional[pd.DataFrame] = None,
        y_df: Optional[pd.DataFrame] = None,
        subsample: bool = False,
    ) -> bool:
        """Check that this feature satisfies the expected feature API"""
        if X_df is None or y_df is None:
            _X_df, _y_df = self.api.load_data()
        if X_df is None:
            X_df = _X_df
        if y_df is None:
            y_df = _y_df
        return validate_feature_api(feature, X_df, y_df, subsample)

    def validate_feature_acceptance(
        self,
        feature: Feature,
        X_df: Optional[pd.DataFrame] = None,
        y_df: Optional[pd.DataFrame] = None,
        subsample: bool = False
    ) -> bool:
        """Evaluate the performance of this feature"""
        if subsample:
            X_df, y_df = subsample_data_for_validation(X_df, y_df)

        result = self.api.engineer_features(X_df, y_df)

        # load accepter for this project
        Accepter = _load_class(self.project, 'validation.feature_accepter')
        accepter = Accepter(result.X_df, result.y, result.features, feature)
        return accepter.judge()


b = Client()
