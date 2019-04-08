import os

from funcy import decorator, ignore

from ballet.exc import (
    ConfigurationError, FeatureRejected, InvalidFeatureApi,
    InvalidProjectStructure, SkippedValidationTest)
from ballet.project import Project
from ballet.util.log import logger, stacklog
from ballet.validation.common import (
    get_accepted_features, get_proposed_feature)
from ballet.validation.feature_acceptance.validator import (
    GFSSFAcceptanceEvaluator)
from ballet.validation.feature_api.validator import FeatureApiValidator
from ballet.validation.feature_pruning.validator import GFSSFPruningEvaluator
from ballet.validation.project_structure.validator import (
    ProjectStructureValidator)

TEST_TYPE_ENV_VAR = 'BALLET_TEST_TYPE'


@decorator
def validation_stage(call, message):
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


def detect_target_type():
    if TEST_TYPE_ENV_VAR in os.environ:
        return os.environ[TEST_TYPE_ENV_VAR]
    else:
        raise ConfigurationError(
            'Could not detect test target type: '
            'missing environment variable {envvar}'
            .format(envvar=TEST_TYPE_ENV_VAR))


@validation_stage('checking project structure')
def check_project_structure(project, force=False):
    if not force and not project.on_pr():
        raise SkippedValidationTest('Not on PR')

    validator = ProjectStructureValidator(project)
    result = validator.validate()
    if not result:
        raise InvalidProjectStructure


@validation_stage('validating feature API')
def validate_feature_api(project, force=False):
    """Validate feature API"""
    if not force and not project.on_pr():
        raise SkippedValidationTest('Not on PR')

    validator = FeatureApiValidator(project)
    result = validator.validate()
    if not result:
        raise InvalidFeatureApi


@validation_stage('evaluating feature performance')
def evaluate_feature_performance(project, force=False):
    """Evaluate feature performance"""
    if not force and not project.on_pr():
        raise SkippedValidationTest('Not on PR')

    out = project.build()
    X_df, y, features = out['X_df'], out['y'], out['features']

    proposed_feature = get_proposed_feature(project)
    accepted_features = get_accepted_features(features, proposed_feature)
    evaluator = GFSSFAcceptanceEvaluator(X_df, y, accepted_features)
    accepted = evaluator.judge(proposed_feature)

    if not accepted:
        raise FeatureRejected


@validation_stage('pruning existing features')
def prune_existing_features(project, force=False):
    """Prune existing features"""
    if not force and not project.on_master_after_merge():
        raise SkippedValidationTest('Not on master')

    out = project.build()
    X_df, y, features = out['X_df'], out['y'], out['features']
    proposed_feature = get_proposed_feature(project)
    accepted_features = get_accepted_features(features, proposed_feature)
    evaluator = GFSSFPruningEvaluator(X_df, y, accepted_features, proposed_feature)
    redundant_features = evaluator.prune()

    # propose removal
    for feature in redundant_features:
        logger.debug('Would prune feature {feature!s}'.format(feature=feature))

    return redundant_features


def validate(package, test_target_type=None):
    """Entrypoint for ./validate.py script in ballet projects"""
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
