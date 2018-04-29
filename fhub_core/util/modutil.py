import importlib
import logging
import pathlib
import pkgutil

logger = logging.getLogger(__name__)


def import_module_from_modname(modname):
    '''Import module'''
    return importlib.import_module(modname)


def import_module_from_relpath(project_root, path):
    '''Import module at relative path to project root'''
    modname = relpath_to_modname(project_root, path)
    return import_module_from_modname(modname)


def import_module_at_path(modname, modpath):
    '''Import module from path that may not be on system path'''
    modpath = pathlib.Path(modpath)
    parentpath = str(modpath.parent)
    modpath = str(modpath)
    importer = pkgutil.get_importer(parentpath)
    mod = importer.find_module(modname).load_module(modname)
    return mod


def relpath_to_modname(relpath):
    parts = pathlib.Path(relpath).parts
    if parts[-1] == '__init__.py':
        parts = parts[:-1]
    elif parts[-1].endswith('.py'):
        parts = list(parts)
        parts[-1] = parts[-1].replace('.py', '')
    else:
        msg = 'Cannot convert a non-python file to a modname'
        msg_detail = 'The relpath given is: {}'.format(relpath)
        logger.error(msg + '\n' + msg_detail)
        raise ValueError(msg)

    return '.'.join(parts)


def modname_to_relpath(project_root, modname, add_init=True):
    '''Convert module name to relative path

    Example:
        >>> modname_to_relpath('dengue_prediction.features')
        'dengue_prediction/features/__init__.py'

    '''
    parts = modname.split('.')
    relpath = pathlib.Path('.').joinpath(*parts)
    if project_root.joinpath(relpath).is_dir():
        if add_init:
            relpath = relpath.joinpath('__init__.py')
    else:
        relpath = relpath + '.py'
    return str(relpath)
