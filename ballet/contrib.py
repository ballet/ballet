import pkgutil
import types
from importlib.abc import PathEntryFinder
from types import ModuleType
from typing import Iterator, List, Optional, cast

from funcy import collecting, notnone

import ballet
from ballet.feature import Feature
from ballet.util import dfilter
from ballet.util.log import logger

__all__ = (
    'collect_contrib_features',
)


def collect_contrib_features(
    project: 'ballet.project.Project'
) -> List[Feature]:
    """Collect contributed features for a project at project_root

    For a project ``foo``, walks modules within the ``foo.features.contrib``
    subpackage. A single object that is an instance of ``ballet.Feature`` is
    imported if present in each module. The resulting ``Feature`` objects are
    collected.

    Args:
        project: project object

    Returns:
        collected features
    """
    contrib = project.resolve('features.contrib')
    return _collect_contrib_features(contrib)


@dfilter(notnone)
@collecting
def _collect_contrib_features(
    module: ModuleType
) -> Iterator[Optional[Feature]]:
    """Collect contributed features from within given module

    Be very careful with untrusted code. The module/package will be
    walked, every submodule will be imported, and all the code therein will be
    executed. But why would you be trying to import from an untrusted package
    anyway?

    Args:
        module: module (standalone or package) that contains feature
            definitions
    """

    if isinstance(module, types.ModuleType):
        # any module that has a __path__ attribute is also a package
        if hasattr(module, '__path__'):
            yield from _collect_contrib_features_from_package(module)
        else:
            yield _collect_contrib_feature_from_module(module)
    else:
        raise ValueError('Input is not a module')


@collecting
def _collect_contrib_features_from_package(
    package: ModuleType
) -> Iterator[Optional[Feature]]:
    logger.debug(
        f'Walking package path {package.__path__} to detect modules...')  # type: ignore  # mypy issue 1422  # noqa E501

    for importer, modname, _ in pkgutil.walk_packages(
            path=package.__path__,  # type: ignore  # mypy issue #1422
            prefix=package.__name__ + '.',
            onerror=logger.error):

        # mistakenly typed as MetaPathFinder
        importer = cast(PathEntryFinder, importer)

        try:
            if importer is None:
                raise ImportError
            # TODO use find_spec
            # https://docs.python.org/3/library/importlib.html#importlib.abc.PathEntryFinder.find_spec)
            finder = importer.find_module(modname)
            if finder is None:
                raise ImportError
            mod = finder.load_module(modname)
        except ImportError:
            logger.exception(f'Failed to import module {modname}')
            continue

        yield _collect_contrib_feature_from_module(mod)


def _collect_contrib_feature_from_module(mod: ModuleType) -> Optional[Feature]:
    logger.debug(
        f'Trying to import contributed feature from module {mod.__name__}...')

    candidates = []
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if isinstance(obj, Feature):
            candidates.append(obj)

    if len(candidates) == 1:
        feature = candidates[0]
        feature.source = mod.__name__
        logger.debug(
            f'Imported 1 feature from {mod.__name__} from {Feature.__name__}'
            ' object')
        return feature
    elif len(candidates) > 1:
        logger.debug(
            f'Found too many {Feature.__name__} objects in module '
            '{mod.__name__}, skipping; candidates were {candidates!r}')
        return None
    else:
        logger.debug(
            f'Failed to import anything useful from module {mod.__name__}')
        return None
