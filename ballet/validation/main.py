from typing import Callable, List

from funcy import decorator, ignore
from stacklog import stacklog

from ballet.exc import (
    FeatureRejected, InvalidFeatureApi, InvalidProjectStructure,
    NoFeaturesCollectedError, SkippedValidationTest)
from ballet.feature import Feature
from ballet.project import Project
from ballet.util.log import logger
from ballet.util.mod import import_module_from_modname
from ballet.validation.common import (
    get_accepted_features, get_proposed_feature)

# helpful for log parsing
PRUNER_MESSAGE = 'Found Redundant Feature: '


@decorator
def validation_stage(call: Callable, message: str):
    call = stacklog(logger.info,
                    'Ballet Validation: {message}'.format(message=message),
                    conditions=[(SkippedValidationTest, 'SKIPPED')])(call)
    call = ignore(SkippedValidationTest)(call)
    return call()


def _load_class(project: Project, config_key: str) -> type:
    path = project.config.get(config_key)
    modname, clsname = path.rsplit('.', maxsplit=1)
    mod = import_module_from_modname(modname)
    cls = getattr(mod, clsname)

    clsname = getattr(cls, '__name__', '<unknown>')
    modfile = getattr(mod, '__file__', '<unknown>')
    logger.debug('Loaded class {} from {}'.format(clsname, modfile))

    return cls


@validation_stage('checking project structure')
def _check_project_structure(project: Project, force: bool = False):
    if not force and not project.on_pr:
        raise SkippedValidationTest('Not on PR')

    Validator = _load_class(project, 'validation.project_structure_validator')
    validator = Validator(project)
    result = validator.validate()
    if not result:
        raise InvalidProjectStructure


@validation_stage('validating feature API')
def _validate_feature_api(project: Project, force: bool = False):
    """Validate feature API"""
    if not force and not project.on_pr:
        raise SkippedValidationTest('Not on PR')

    Validator = _load_class(project, 'validation.feature_api_validator')
    validator = Validator(project)
    result = validator.validate()
    if not result:
        raise InvalidFeatureApi


@validation_stage('evaluating feature performance')
def _evaluate_feature_performance(project: Project, force: bool = False):
    """Evaluate feature performance"""
    if not force and not project.on_pr:
        raise SkippedValidationTest('Not on PR')

    result = project.api.engineer_features()
    X_df, y, features = result.X_df, result.y, result.features

    proposed_feature = get_proposed_feature(project)
    accepted_features = get_accepted_features(features, proposed_feature)

    Accepter = _load_class(project, 'validation.feature_accepter')
    accepter = Accepter(X_df, y, accepted_features, proposed_feature)
    accepted = accepter.judge()

    if not accepted:
        raise FeatureRejected


@validation_stage('pruning existing features')
def _prune_existing_features(
    project: Project, force: bool = False
) -> List[Feature]:
    """Prune existing features"""
    if not force and not project.on_master_after_merge:
        raise SkippedValidationTest('Not on master')

    try:
        proposed_feature = get_proposed_feature(project)
    except NoFeaturesCollectedError:
        raise SkippedValidationTest('No features collected')

    result = project.api.engineer_features()
    X_df, y, features = result.X_df, result.y, result.features
    accepted_features = get_accepted_features(features, proposed_feature)

    Pruner = _load_class(project, 'validation.feature_pruner')
    pruner = Pruner(X_df, y, accepted_features, proposed_feature)
    redundant_features = pruner.prune()

    # "propose removal"
    for feature in redundant_features:
        logger.info(PRUNER_MESSAGE + feature.source)

    return redundant_features


def validate(project: Project,
             check_project_structure: bool,
             check_feature_api: bool,
             evaluate_feature_acceptance: bool,
             evaluate_feature_pruning: bool):
    """Entrypoint for 'ballet validate' command in ballet projects"""
    if check_project_structure:
        _check_project_structure(project)
    if check_feature_api:
        _validate_feature_api(project)
    if evaluate_feature_acceptance:
        _evaluate_feature_performance(project)
    if evaluate_feature_pruning:
        _prune_existing_features(project)
