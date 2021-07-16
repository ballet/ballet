import pathlib
from copy import deepcopy
from typing import Optional

import numpy as np

from ballet.encoder import EncoderPipeline, make_robust_encoder
from ballet.eng import BaseTransformer
from ballet.pipeline import FeatureEngineeringPipeline
from ballet.project import Project
from ballet.util.mod import import_module_from_modname

PRIMITIVES_PATH = [pathlib.Path(__file__).with_name('primitives').resolve()]
PIPELINES_PATH = [pathlib.Path(__file__).with_name('pipelines').resolve()]

_project_detection_details = """\
The primitive detects the Ballet project in one of three ways. First, if
the package slug is given, then Ballet will import the package and detect
the project. Second, if a path to the project is given (i.e. the directory
containing the ballet.yml configuration file), then the project will be
detected that way. Third, if neither is given, then Ballet will try to
detect a project in the current working directory. This third option is the
least robust and sensitive to where you are running the MLPipeline from.

Args:
    package_slug: name of top-level package for the Ballet project
    project_path: filesystem path to the directory containing a valid
        ballet.yml configuration file
"""


def _get_project(package_slug, project_path):
    if package_slug is not None:
        package = import_module_from_modname(package_slug)
        return Project(package)
    elif project_path is not None:
        return Project.from_path(project_path)
    else:
        return Project.from_cwd()


def make_engineer_features(
    package_slug: Optional[str] = None,
    project_path: Optional[str] = None
) -> FeatureEngineeringPipeline:
    f"""Make the engineer_features primitive for the given Ballet project

    {_project_detection_details}

    Returns:
        a deep copy of the feature engineering pipeline defined by the project
    """
    project = _get_project(package_slug, project_path)
    return deepcopy(project.api.pipeline)


def make_encode_target(
    package_slug: Optional[str] = None,
    project_path: Optional[str] = None
) -> EncoderPipeline:
    f"""Make the encode_target primitive for the given Ballet project

    {_project_detection_details}

    Returns:
        a deep copy of the target encoder pipeline defined by the project
    """
    project = _get_project(package_slug, project_path)
    encoder = deepcopy(project.api.encoder)
    return make_robust_encoder(encoder, can_skip_transform_none=True)


class DropMissingTargets(BaseTransformer):

    def fit(self, X, y, **fit_kwargs):
        self.inds_ = ~np.isnan(y)

    def transform(self, X, y):
        if y is not None:
            if hasattr(X, 'loc'):
                return X.loc(axis=0)[self.inds_], y[self.inds_]
            else:
                return X[self.inds_, :], y[self.inds_]
        else:
            return X, y
