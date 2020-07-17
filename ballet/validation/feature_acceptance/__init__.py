from types import ModuleType
from typing import Optional

import pandas as pd

from ballet.feature import Feature
from ballet.project import Project
from ballet.util.typing import Pathy
from ballet.validation.common import subsample_data_for_validation
from ballet.validation.main import _load_class


def validate_feature_acceptance(
    feature: Feature,
    X: pd.DataFrame,
    y: pd.DataFrame,
    subsample: bool = False,
    path: Optional[Pathy] = None,
    package: Optional[ModuleType] = None
):
    if package is not None:
        project = Project(package)
    elif path is not None:
        project = Project.from_path(path)
    else:
        project = Project.from_cwd()

    if subsample:
        X, y = subsample_data_for_validation(X, y)

    # build project
    result = project.api.engineer_features(X, y)

    # load accepter for this project
    Accepter = _load_class(project, 'validation.feature_accepter')
    accepter = Accepter(result.X_df, result.y, result.features, feature)
    return accepter.judge()
