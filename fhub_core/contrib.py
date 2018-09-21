import importlib
import pkgutil
import types

import funcy

from ballet.feature import Feature
from ballet.util.log import logger

__all__ = [
    'get_contrib_features',
]


def get_contrib_features(contrib):
    '''Get contributed features from within given module

    Be very careful with untrusted code. The module/package will be
    walked, every submodule will be imported, and all the code therein will be
    executed. But why would you be trying to import from an untrusted package
    anyway?

    Args:
        contrib (module): module (standalone or package) that contains feature
            definitions

    Returns:
        List of Feature
    '''

    if isinstance(contrib, types.ModuleType):
        contrib_features = []

        # fuuuuu
        importlib.invalidate_caches()

        # any module that has a __path__ attribute is a package
        if hasattr(contrib, '__path__'):
            features = get_contrib_features_from_package(contrib)
            contrib_features.extend(features)
        else:
            features = get_contrib_features_from_module(contrib)
            contrib_features.extend(features)
        return contrib_features
    else:
        raise ValueError('Input is not a module')


def get_contrib_features_from_package(package):
    contrib_features = []

    logger.debug(
        'Walking package path {path} to detect modules...'
        .format(path=package.__path__))
    for importer, modname, _ in pkgutil.walk_packages(
            path=package.__path__,
            prefix=package.__name__ + '.',
            onerror=logger.error):
        try:
            mod = importer.find_module(modname).load_module(modname)
        except ImportError:
            logger.exception(
                'Failed to import module {modname}'
                .format(modname=modname))
            continue
        features = get_contrib_features_from_module(mod)
        contrib_features.extend(features)

    return contrib_features


def get_contrib_features_from_module(mod):
    contrib_features = []

    logger.debug(
        'Trying to import contributed feature(s) from module {modname}...'
        .format(modname=mod.__name__))

    # case 1: file defines `features` variable
    try:
        features = import_contrib_feature_from_collection(mod)
        contrib_features.extend(features)
        logger.debug(
            'Imported {n} feature(s) from {modname} from collection'
            .format(n=len(features), modname=mod.__name__))
    except ImportError:
        # case 2: file has at least `input` and `transformer` defined
        try:
            feature = import_contrib_feature_from_components(mod)
            contrib_features.append(feature)
            logger.debug(
                'Imported 1 feature from {modname} from components'
                .format(modname=mod.__name__))
        except ImportError:
            # case 3: nothing useful in file
            logger.debug(
                'Failed to import anything useful from module {modname}'
                .format(modname=mod.__name__))

    return contrib_features


def import_contrib_feature_from_components(mod):
    required = ['input', 'transformer']
    optional = ['name', 'description', 'output', 'options']
    required_vars, optional_vars = import_names_from_module(
        mod, required, optional)
    feature = Feature(
        input=required_vars['input'],
        transformer=required_vars['transformer'],
        source=mod.__name__,
        **optional_vars)
    return feature


def import_contrib_feature_from_collection(mod):
    required = 'features'
    optional = None
    required_vars, _ = import_names_from_module(
        mod, required, optional)
    features = required_vars['features']
    for feature in features:
        feature.source = mod.__name__
    return features


def import_names_from_module(mod, required, optional):

    msg = funcy.partial(
        'Required variable {varname} not found in module {modname}'
        .format, modname=mod.__name__)

    # required vars
    if required:
        required_vars = {}
        if isinstance(required, str):
            required = [required]
        for varname in required:
            if hasattr(mod, varname):
                required_vars[varname] = getattr(mod, varname)
            else:
                raise ImportError(msg(varname=varname))
    else:
        required_vars = None

    # optional vars
    if optional:
        if isinstance(optional, str):
            optional = [optional]
        optional_vars = {k: getattr(mod, k)
                         for k in optional if hasattr(mod, k)}
    else:
        optional_vars = None

    return required_vars, optional_vars
