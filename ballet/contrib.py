import importlib
import pkgutil
import types

from ballet.feature import Feature
from ballet.project import Project
from ballet.util.log import logger

__all__ = [
    'get_contrib_features',
]


def get_contrib_features(root):
    project = Project(root)
    contrib = project._resolve('.features.contrib')
    return _get_contrib_features(contrib)


def _get_contrib_features(module):
    '''Get contributed features from within given module

    Be very careful with untrusted code. The module/package will be
    walked, every submodule will be imported, and all the code therein will be
    executed. But why would you be trying to import from an untrusted package
    anyway?

    Args:
        contrib (module): module (standalone or package) that contains feature
            definitions

    Returns:
        List[Feature]: list of features
    '''

    if isinstance(module, types.ModuleType):
        contrib_features = []

        # fuuuuu
        importlib.invalidate_caches()

        # any module that has a __path__ attribute is a package
        if hasattr(module, '__path__'):
            features = _get_contrib_features_from_package(module)
            contrib_features.extend(features)
        else:
            features = _get_contrib_features_from_module(module)
            contrib_features.extend(features)
        return contrib_features
    else:
        raise ValueError('Input is not a module')


def _get_contrib_features_from_package(package):
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
        features = _get_contrib_features_from_module(mod)
        contrib_features.extend(features)

    return contrib_features


def _get_contrib_features_from_module(mod):
    contrib_features = []

    logger.debug(
        'Trying to import contributed feature(s) from module {modname}...'
        .format(modname=mod.__name__))

    try:
        feature = import_contrib_feature_from_feature(mod)
        contrib_features.append(feature)
        logger.debug(
            'Imported 1 feature from {modname} from Feature object'
            .format(modname=mod.__name__))
    except ImportError:
        logger.debug(
            'Failed to import anything useful from module {modname}'
            .format(modname=mod.__name__))

    return contrib_features


def import_contrib_feature_from_feature(mod):
    candidates = []
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if isinstance(obj, Feature):
            candidates.append(obj)

    if len(candidates) == 1:
        feature = candidates[0]
        feature.source = mod.__name__
        return feature
    elif len(candidates) > 1:
        raise ImportError(
            'Found too many \'Feature\' objects in module {modname}: '
            '{candidates!r}'
            .format(modname=mod.__name__, candidates=candidates))
    else:
        raise ImportError(
            'Did not find any \'Feature\' objects in module {modname}'
            .format(modname=mod.__name__))
