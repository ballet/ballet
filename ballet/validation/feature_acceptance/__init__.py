from ballet.project import Project
from ballet.validation.common import subsample_data_for_validation
from ballet.validation.main import _load_class


def validate_feature_acceptance(feature, X, y, subsample=False, path=None,
                                package=None):
    if package is not None:
        project = Project(package)
    elif path is not None:
        project = Project.from_path(path)
    else:
        project = Project.from_cwd()

    if subsample:
        X, y = subsample_data_for_validation(X, y)

    # build project
    result = project.build(X, y)

    # load accepter for this project
    Accepter = _load_class(project, 'validation.feature_accepter')
    accepter = Accepter(result.X_df, result.y, result.features, feature)
    return accepter.judge()
