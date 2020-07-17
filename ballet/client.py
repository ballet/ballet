from os import PathLike
from types import ModuleType
from typing import Optional

import pandas as pd
from funcy import cached_property

from ballet.feature import Feature
from ballet.project import Project
from ballet.validation.common import subsample_data_for_validation
from ballet.validation.feature_api import validate_feature_api
from ballet.validation.main import _load_class


class Client:

    def __init__(self, prj=None):
        self._prj = prj

    @cached_property
    def _project(self) -> Project:
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

    def validate_feature_api(
        self,
        feature: Feature,
        X_df: Optional[pd.DataFrame] = None,
        y_df: Optional[pd.DataFrame] = None,
        subsample=False,
    ) -> bool:
        if X_df is None or y_df is None:
            _X_df, _y_df = self._project.api.load_data()
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
        project = self._project

        if subsample:
            X_df, y_df = subsample_data_for_validation(X_df, y_df)

        result = project.api.engineer_features(X_df, y_df)

        # load accepter for this project
        Accepter = _load_class(project, 'validation.feature_accepter')
        accepter = Accepter(result.X_df, result.y, result.features, feature)
        return accepter.judge()


b = Client()
