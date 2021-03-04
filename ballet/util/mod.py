import importlib
import pathlib
import pkgutil
from types import ModuleType
from typing import Optional

from ballet.util.log import logger
from ballet.util.typing import Pathy


def import_module_from_modname(modname: str) -> ModuleType:
    """Import module"""
    return importlib.import_module(modname)


def import_module_from_relpath(path: Pathy) -> ModuleType:
    """Import module at relative path to project root"""
    modname = relpath_to_modname(path)
    return import_module_from_modname(modname)


def import_module_at_path(modname: str, modpath: Pathy) -> ModuleType:
    """Import module from path that may not be on system path

    Args:
        modname: module name from package root, e.g. foo.bar
        modpath: absolute path to module itself,
            e.g. /home/user/foo/bar.py. In the case of a module that is a
            package, then the path should be specified as '/home/user/foo' and
            a file '/home/user/foo/__init__.py' *must be present* or the import
            will fail.

    Examples:
        >>> modname = 'foo.bar.baz'
        >>> modpath = '/home/user/foo/bar/baz.py'
        >>> import_module_at_path(modname, modpath)
        <module 'foo.bar.baz' from '/home/user/foo/bar/baz.py'>

        >>> modname = 'foo.bar'
        >>> modpath = '/home/user/foo/bar'
        >>> import_module_at_path(modname, modpath)
        <module 'foo.bar' from '/home/user/foo/bar/__init__.py'>
    """
    # TODO just keep navigating up in the source tree until an __init__.py is
    # not found?

    modpath = pathlib.Path(modpath).resolve()

    if modpath.name == '__init__.py':
        # TODO improve debugging output with recommend change
        raise ValueError('Don\'t provide the __init__.py!')

    def is_package(modpath):
        return modpath.suffix != '.py'

    def has_init(dir):
        return dir.joinpath('__init__.py').is_file()

    def has_package_structure(modname, modpath):
        modparts = modname.split('.')
        n = len(modparts)
        dir = modpath
        if not is_package(modpath):
            n = n - 1
            dir = dir.parent
        while n > 0:
            if not has_init(dir):
                return False
            dir = dir.parent
            n = n - 1

        return True

    if not has_package_structure(modname, modpath):
        raise ImportError('Module does not have valid package structure.')

    parentpath = str(pathlib.Path(modpath).parent)

    finder = pkgutil.get_importer(parentpath)
    if finder is None:
        raise ImportError(
            f'Failed to find loader for module {modname} within '
            f'dir {parentpath}')

    loader = finder.find_module(modname)
    if loader is None:
        raise ImportError(
            f'Failed to find loader for module {modname} within '
            f'dir {parentpath}')

    mod = loader.load_module(modname)

    # TODO figure out what to do about this
    assert mod.__name__ == modname

    return mod


def relpath_to_modname(relpath: Pathy) -> str:
    """Convert relative path to module name

    Within a project, a path to the source file is uniquely identified with a
    module name. Relative paths of the form 'foo/bar' are *not* converted to
    module names 'foo.bar', because (1) they identify directories, not regular
    files, and (2) already 'foo/bar/__init__.py' would claim that conversion.

    Args:
        relpath: Relative path from some location on sys.path

    Example:
        >>> relpath_to_modname('ballet/util/_util.py')
        'ballet.util._util'
    """
    # don't try to resolve!
    p = pathlib.Path(relpath)

    if p.name == '__init__.py':
        p = p.parent
    elif p.suffix == '.py':
        p = p.with_suffix('')
    else:
        msg = 'Cannot convert a non-python file to a modname'
        msg_detail = f'The relpath given is: {relpath}'
        logger.error(msg + '\n' + msg_detail)
        raise ValueError(msg)

    return '.'.join(p.parts)


def modname_to_relpath(
    modname: str,
    project_root: Optional[Pathy] = None,
    add_init: bool = True,
) -> str:
    """Convert module name to relative path.

    The project root is usually needed to detect if the module is a package, in
    which case the relevant file is the `__init__.py` within the subdirectory.

    Example:
        >>> modname_to_relpath('foo.features')
        'foo/features.py'
        >>> modname_to_relpath('foo.features',
                               project_root='/path/to/project')
        'foo/features/__init__.py'

    Args:
        modname: Module name, e.g. `os.path`
        project_root: Path to project root
        add_init: Whether to add `__init__.py` to the path of modules
            that are packages. Defaults to True
    """
    parts = modname.split('.')
    relpath = pathlib.Path(*parts)

    # is the module a package? if so, the relpath identifies a directory
    # it is easier to check for whether a file is a directory than to try to
    # import the module dynamically and see whether it is a package
    if project_root is not None:
        relpath_resolved = pathlib.Path(project_root).joinpath(relpath)
    else:
        relpath_resolved = relpath

    is_dir = relpath_resolved.is_dir()
    if is_dir and add_init:
        result = relpath.joinpath('__init__.py')
    elif is_dir and not add_init:
        result = relpath
    else:
        result = relpath.with_suffix('.py')
    return str(result)
