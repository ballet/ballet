from typing import Callable, List

from funcy import decorator, func_partial, ignore
from funcy.decorators import Call
from stacklog import stacklog

from ballet.exc import (
    FeatureRejected, InvalidFeatureApi, InvalidProjectStructure,
    NoFeaturesCollectedError, SkippedValidationTest,)
from ballet.feature import Feature
from ballet.project import Project
from ballet.util.log import logger
from ballet.util.mod import import_module_from_modname
from ballet.validation.common import (
    get_accepted_features, get_proposed_feature,)

# helpful for log parsing
PRUNER_MESSAGE = 'Found Redundant Feature: '


@decorator
def validation_stage(call: Call, message: str):
    call = stacklog(logger.info,
                    f'Ballet Validation: {message}',
                    conditions=[(SkippedValidationTest, 'SKIPPED')])(call)
    call = ignore(SkippedValidationTest)(call)
    return call()


def _load_validator_class_params(
    project: Project, config_key: str
) -> Callable:
    """Load validator class according to config_key with optional params

    At the provided key, the config should show an entry in one of two forms:
    1. The fully-qualified class name of the validator (str)
    2. A yaml hash with the key `name` mapping to the fully-qualified class
    name of the validator, and optionally, the key `params` mapping to hash
    of keyword arguments to be passed to the validator class.

    If `params` is provided, then they are partially applied to the
    validator class ``__init__`` method such that calls to create an
    instance of the validator class have the given params set as keyword
    arguments.

    For example, if the yaml file looks like::

       foo:
         bar:
           validation:
               feature_accepter:
                 name: baz.qux.MyFeatureAccepter
                 params:
                   key1: value1

    Then::

       make_validator = _load_validator_class_params(project, 'foo.bar.validation.feature_accepter')

    would result in the following equivalence::

       make_validator(arg)
       baz.qux.MyFeatureAccepter(arg, key1=value1)
    """  # noqa E501
    entry = project.config.get(config_key)
    if isinstance(entry, str):
        path = entry
        params = {}
    else:
        path = entry.get('name')
        params = dict(entry.get('params'))

    modname, clsname = path.rsplit('.', maxsplit=1)
    mod = import_module_from_modname(modname)
    cls = getattr(mod, clsname)
    clsname = getattr(cls, '__name__', '<unknown>')
    modfile = getattr(mod, '__file__', '<unknown>')
    logger.debug(
        'Loaded class %s from module at %s with params %r',
        clsname, modfile, params)
    return func_partial(cls, **params)


@validation_stage('checking project structure')
def _check_project_structure(project: Project, force: bool = False):
    if not force and project.on_master:
        raise SkippedValidationTest('Not on feature branch')

    validator_class = _load_validator_class_params(
        project, 'validation.project_structure_validator')
    validator = validator_class(project)
    result = validator.validate()
    if not result:
        raise InvalidProjectStructure


@validation_stage('validating feature API')
def _validate_feature_api(project: Project, force: bool = False):
    """Validate feature API"""
    if not force and project.on_master:
        raise SkippedValidationTest('Not on feature branch')

    validator_class = _load_validator_class_params(
        project, 'validation.feature_api_validator')
    validator = validator_class(project)
    result = validator.validate()
    if not result:
        raise InvalidFeatureApi


@validation_stage('evaluating feature performance')
def _evaluate_feature_performance(project: Project, force: bool = False):
    """Evaluate feature performance"""
    if not force and project.on_master:
        raise SkippedValidationTest('Not on feature branch')

    result = project.api.engineer_features()
    X_df, y_df, y = result.X_df, result.y_df, result.y
    features = result.features

    proposed_feature = get_proposed_feature(project)
    accepted_features = get_accepted_features(features, proposed_feature)

    accepter_class = _load_validator_class_params(
        project, 'validation.feature_accepter')
    accepter = accepter_class(
        X_df, y_df, y, accepted_features, proposed_feature)
    accepted = accepter.judge()

    if not accepted:
        raise FeatureRejected


@validation_stage('pruning existing features')
def _prune_existing_features(
    project: Project, force: bool = False
) -> List[Feature]:
    """Prune existing features"""
    if not force and not project.on_master:
        raise SkippedValidationTest('Not on master')

    try:
        # if on master but not after merge, then we diff master with itself
        # and collect no features.
        proposed_feature = get_proposed_feature(project)
    except NoFeaturesCollectedError:
        raise SkippedValidationTest('No features collected')

    result = project.api.engineer_features()
    X_df, y_df, y = result.X_df, result.y_df, result.y
    features = result.features
    accepted_features = get_accepted_features(features, proposed_feature)

    pruner_class = _load_validator_class_params(
        project, 'validation.feature_pruner')
    pruner = pruner_class(X_df, y_df, y, accepted_features, proposed_feature)
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
