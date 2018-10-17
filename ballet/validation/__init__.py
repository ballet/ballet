import os

from funcy import decorator, ignore

from ballet.contrib import _get_contrib_feature_from_module
from ballet.exc import (
    Error, ConfigurationError, FeatureRejected, InvalidFeatureApi,
    InvalidProjectStructure, SkippedValidationTest)
from ballet.project import Project
from ballet.util.log import logger, stacklog
from ballet.validation.alpha_investing import (
    AlphaInvestingAcceptanceEvaluator, load_alpha, save_alpha)
from ballet.validation.feature_evaluation import NoOpPruningEvaluator
from ballet.validation.project_structure import (
    ChangeCollector, FeatureApiValidator, FileChangeValidator)


TEST_TYPE_ENV_VAR = 'BALLET_TEST_TYPE'


@decorator
def log_validation_stage(call, message):
    call = stacklog(logger.info,
                    'Ballet Validation: {message}'.format(message=message),
                    conditions=[(SkippedValidationTest, 'SKIPPED')])(call)
    call = ignore(SkippedValidationTest)(call)
    return call()


class BalletTestTypes:
    PROJECT_STRUCTURE_VALIDATION = 'project_structure_validation'
    FEATURE_API_VALIDATION = 'feature_api_validation'
    FEATURE_ACCEPTANCE_EVALUTION = 'feature_acceptance_evaluation'
    FEATURE_PRUNING_EVALUATION = 'feature_pruning_evaluation'


def get_proposed_feature(project):
    change_collector = ChangeCollector(project)
    collected_changes = change_collector.collect_changes()
    if len(collected_changes.new_feature_info) != 1:
        raise Error
    importer, _, _ = collected_changes.new_feature_info[0]
    module = importer()
    feature = _get_contrib_feature_from_module(module)
    return feature


def detect_target_type():
    if TEST_TYPE_ENV_VAR in os.environ:
        return os.environ[TEST_TYPE_ENV_VAR]
    else:
        raise ConfigurationError(
            'Could not detect test target type: '
            'missing environment variable {envvar}'
            .format(envvar=TEST_TYPE_ENV_VAR))


@log_validation_stage('checking project structure')
def check_project_structure(project):
    if not project.on_pr():
        raise SkippedValidationTest

    validator = FileChangeValidator(project)
    result = validator.validate()
    if not result:
        raise InvalidProjectStructure


@log_validation_stage('validating feature API')
def validate_feature_api(project):
    if not project.on_pr():
        raise SkippedValidationTest

    validator = FeatureApiValidator(project)
    result = validator.validate()
    if not result:
        raise InvalidFeatureApi


@log_validation_stage('evaluating feature performance')
def evaluate_feature_performance(project):
    if not project.on_pr():
        raise SkippedValidationTest

    out = project.build()
    X_df, y, features = out['X_df'], out['y'], out['features']

    ai, i = load_alpha(project)

    evaluator = AlphaInvestingAcceptanceEvaluator(X_df, y, features, ai)
    proposed_feature = get_proposed_feature(project)
    accepted = evaluator.judge(proposed_feature)

    save_alpha(project, ai, i, accepted)

    if not accepted:
        raise FeatureRejected


@log_validation_stage('pruning existing features')
def prune_existing_features(project):
    if project.on_pr():
        raise SkippedValidationTest

    out = project.build()
    X_df, y, features = out['X_df'], out['y'], out['features']
    evaluator = NoOpPruningEvaluator(X_df, y, features)
    redundant_features = evaluator.prune()

    # propose removal
    for feature in redundant_features:
        pass


def main(package, test_target_type=None):
    project = Project(package)

    if test_target_type is None:
        test_target_type = detect_target_type()

    if test_target_type == BalletTestTypes.PROJECT_STRUCTURE_VALIDATION:
        check_project_structure(project)
    elif test_target_type == BalletTestTypes.FEATURE_API_VALIDATION:
        validate_feature_api(project)
    elif test_target_type == BalletTestTypes.FEATURE_ACCEPTANCE_EVALUTION:
        evaluate_feature_performance(project)
    elif test_target_type == (
            BalletTestTypes.FEATURE_PRUNING_EVALUATION):
        prune_existing_features(project)
    else:
        raise NotImplementedError(
            'Unsupported test target type: {test_target_type}'
            .format(test_target_type=test_target_type))
