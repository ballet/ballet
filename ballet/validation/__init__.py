import os

from ballet.exc import (
    ConfigurationError, FeatureRejected, InvalidFeatureApi,
    InvalidProjectStructure)
from ballet.project import Project
from ballet.validation.feature_evaluation import (
    FeatureRedundancyEvaluator, FeatureRelevanceEvaluator)
from ballet.validation.project_structure import (
    ChangeCollector, FeatureApiValidator, FileChangeValidator)


TEST_TYPE_ENV_VAR = 'TEST_TYPE'


def get_proposed_features(project):
    change_collector = ChangeCollector(project)
    _, _, _, new_feature_info = change_collector.collect_changes()
    # TODO import features
    return new_feature_info


def detect_target_type():
    if TEST_TYPE_ENV_VAR in os.environ:
        return os.environ[TEST_TYPE_ENV_VAR]
    else:
        raise ConfigurationError(
            'Could not detect test target type: '
            'missing environment variable {envvar}'
            .format(envvar=TEST_TYPE_ENV_VAR))


def check_project_structure(project):
    validator = FileChangeValidator(project)
    result = validator.validate()
    if not result:
        raise InvalidProjectStructure


def validate_feature_api(project):
    validator = FeatureApiValidator(project)
    result = validator.validate()
    if not result:
        raise InvalidFeatureApi


def evaluate_feature_performance(project):
    X_df, y_df = project.load_data()
    features = project.get_contrib_features()
    evaluator = FeatureRelevanceEvaluator(X_df, y_df, features)
    proposed_features = get_proposed_features(project)
    accepted = evaluator.judge(proposed_features)
    if not accepted:
        raise FeatureRejected


def prune_existing_features(project):
    X_df, y_df = project.load_data()
    features = project.get_contrib_features()
    evaluator = FeatureRedundancyEvaluator(X_df, y_df, features)
    redundant_features = evaluator.prune()

    # propose removal
    for feature in redundant_features:
        pass


def main(package, target_type=None):

    project = Project(package)

    if target_type is None:
        target_type = detect_target_type()

    if target_type == 'project_structure_validation':
        check_project_structure(project)
    elif target_type == 'feature_api_validation':
        validate_feature_api(project)
    elif target_type == 'pre_acceptance_feature_evaluation':
        evaluate_feature_performance(project)
    elif target_type == 'post_acceptance_feature_evaluation':
        prune_existing_features(project)
    else:
        raise NotImplementedError
