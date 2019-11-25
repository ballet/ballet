from funcy import decorator, ignore

from ballet.exc import (
    FeatureRejected, InvalidFeatureApi, InvalidProjectStructure,
    SkippedValidationTest)
from ballet.util.log import logger, stacklog
from ballet.util.mod import import_module_from_modname
from ballet.validation.common import (
    get_accepted_features, get_proposed_feature)

# helpful for log parsing
PRUNER_MESSAGE = 'Found Redundant Feature: '


@decorator
def validation_stage(call, message):
    call = stacklog(logger.info,
                    'Ballet Validation: {message}'.format(message=message),
                    conditions=[(SkippedValidationTest, 'SKIPPED')])(call)
    call = ignore(SkippedValidationTest)(call)
    return call()


def load_class(project, config_key):
    path = project.config.get(config_key)
    modname, clsname = path.rsplit('.', maxsplit=1)
    mod = import_module_from_modname(modname)
    cls = getattr(mod, clsname)
    logger.debug('Loaded class {} from {}'
                 .format(cls.__name__, mod.__path__))
    return cls


@validation_stage('checking project structure')
def _check_project_structure(project, force=False):
    if not force and not project.on_pr:
        raise SkippedValidationTest('Not on PR')

    Validator = load_class(project, 'validation.project_structure_validator')
    validator = Validator(project)
    result = validator.validate()
    if not result:
        raise InvalidProjectStructure


@validation_stage('validating feature API')
def _validate_feature_api(project, force=False):
    """Validate feature API"""
    if not force and not project.on_pr:
        raise SkippedValidationTest('Not on PR')

    Validator = load_class(project, 'validation.feature_api_validator')
    validator = Validator(project)
    result = validator.validate()
    if not result:
        raise InvalidFeatureApi


@validation_stage('evaluating feature performance')
def _evaluate_feature_performance(project, force=False):
    """Evaluate feature performance"""
    if not force and not project.on_pr:
        raise SkippedValidationTest('Not on PR')

    out = project.build()
    X_df, y, features = out['X_df'], out['y'], out['features']

    proposed_feature = get_proposed_feature(project)
    accepted_features = get_accepted_features(features, proposed_feature)

    Accepter = load_class(project, 'validation.feature_accepter')
    accepter = Accepter(X_df, y, accepted_features, proposed_feature)
    accepted = accepter.judge()

    if not accepted:
        raise FeatureRejected


@validation_stage('pruning existing features')
def _prune_existing_features(project, force=False):
    """Prune existing features"""
    if not force and not project.on_master_after_merge:
        raise SkippedValidationTest('Not on master')

    out = project.build()
    X_df, y, features = out['X_df'], out['y'], out['features']
    proposed_feature = get_proposed_feature(project)
    accepted_features = get_accepted_features(features, proposed_feature)

    Pruner = load_class(project, 'validation.feature_pruner')
    pruner = Pruner(X_df, y, accepted_features, proposed_feature)
    redundant_features = pruner.prune()

    # "propose removal"
    for feature in redundant_features:
        logger.info(PRUNER_MESSAGE + feature.source)

    return redundant_features


def validate(project,
             check_project_structure,
             check_feature_api,
             evaluate_feature_acceptance,
             evaluate_feature_pruning):
    """Entrypoint for 'ballet validate' command in ballet projects"""
    if check_project_structure:
        _check_project_structure(project)
    if check_feature_api:
        _validate_feature_api(project)
    if evaluate_feature_acceptance:
        _evaluate_feature_performance(project)
    if evaluate_feature_pruning:
        _prune_existing_features(project)
