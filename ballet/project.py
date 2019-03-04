from importlib import import_module

import git
import yaml
from funcy import get_in, memoize, partial

from ballet.compat import pathlib
from ballet.exc import ConfigurationError
from ballet.util.ci import get_travis_pr_num
from ballet.util.git import get_pr_num
from ballet.util.mod import import_module_at_path

DEFAULT_CONFIG_NAME = 'ballet.yml'


def get_config_paths(package_root):
    """Get candidate config paths

    Creates a sequence of paths that includes the package root and all of its
    parents, as well as ~/.ballet.

    Args:
        package_root (path-like): Directory of the ballet project root
            directory, the one usually containing the ``ballet.yml`` file.
    """
    package_root = pathlib.Path(package_root)

    # parents of package directory, and the package root just in case
    paths = [
        d.joinpath(DEFAULT_CONFIG_NAME)
        for d in [package_root] + list(package_root.parents)
    ]

    # home directory
    paths.append(
        pathlib.Path.home().joinpath('.ballet', DEFAULT_CONFIG_NAME))

    # defaults in ballet repo

    return paths


def load_config_at_path(path):
    if path.exists() and path.is_file():
        with path.open('r') as f:
            return yaml.load(f)
    else:
        return None


def load_config_in_dir(path):
    if path.exists() and path.is_dir():
        candidate = path.joinpath(DEFAULT_CONFIG_NAME)
        return load_config_at_path(candidate)


@memoize
def find_configs(package_root):
    """Find valid ballet project config files

    See if any of the candidates returned by get_config_paths are valid.

    Args:
        package_root (path-like): Directory of the ballet project root
            directory, the one usually containing the ``ballet.yml`` file.

    Returns:
        list[tuple]: List of (dict, str) representing config
            information and the path that information was loaded from

    Raises:
        ConfigurationError: No valid config files were found.
    """
    configs = []
    for candidate in get_config_paths(package_root):
        config = load_config_at_path(candidate)
        if config is not None:
            configs.append((config, candidate))

    if configs:
        return configs
    else:
        raise ConfigurationError("Couldn't find any ballet.yml config files.")


def config_get(package_root, *path, default=None):
    """Get a configuration option following a path through the config

    Example usage:

        >>> config_get('/path/to/package',
                       'problem', 'problem_type_details', 'scorer',
                       default='accuracy')

    Args:
        package_root (path-like): Directory of the ballet project root
            directory, the one usually containing the ``ballet.yml`` file.
        *path (list[str]): List of config sections and options to follow.
        default (default=None): A default value to return in the case that
            the option does not exist.
    """
    config_info = find_configs(package_root)

    o = object()
    for config, _ in config_info:
        result = get_in(config, path, default=o)
        if result is not o:
            return result

    return default


def make_config_get(package_root):
    """Return a function to get configuration options for a specific project



    """
    package_root = pathlib.Path(package_root).resolve()
    return partial(config_get, package_root)


def relative_to_contrib(diff, project):
    """Compute relative path of changed file to contrib dir

    Args:
        diff (git.diff.Diff): file diff
        project (Project): project

    Returns:
        Path
    """
    path = pathlib.Path(diff.b_path)
    contrib_path = project.contrib_module_path
    return path.relative_to(contrib_path)


class Project:
    """Encapsulate information on a ballet project

    This is a utility class mostly useful for easy access to the project's
    information from within the ballet.validation package.

    In addition to the defined methods and properties, the following functions
    of the project can be accessed as attributes of a class instance, where
    ``prj`` refers to the python module of the underlying ballet project:
    - ``conf`` (``prj.conf``)
    - ``get`` (``prj.conf.get``)
    - ``load_data`` (``prj.load_data.load_data``)
    - ``build`` (``prj.features.build``)
    - ``get_contrib_features`` (``prj.features.get_contrib_features``)

    Args:
        package (ModuleType): python package representing imported ballet
            project
    """

    attr_map = {
        'conf': ('.conf', None),
        'get': ('.conf', 'get'),
        'load_data': ('.load_data', 'load_data'),
        'build': ('.features', 'build'),
        'get_contrib_features': ('.features', 'get_contrib_features')
    }

    def __init__(self, package):
        self.package = package

    @classmethod
    def from_path(cls, path):
        config = load_config_in_dir(path)
        if config is None:
            raise ConfigurationError

        project_slug = get_in(config, ('project', 'slug'))
        package = import_module_at_path(project_slug,
                                        path.joinpath(project_slug))
        return cls(package)

    def _resolve(self, modname, attr=None):
        module = import_module(modname, package=self.package.__name__)
        if attr is not None:
            return getattr(module, attr)
        else:
            return module

    @property
    def pr_num(self):
        """Return the PR number or None if not on a PR"""
        result = get_pr_num(repo=self.repo)
        if result is None:
            result = get_travis_pr_num()
        return result

    def on_pr(self):
        """Return whether the project has a source tree on a PR"""
        return self.pr_num is not None

    @property
    def path(self):
        """Return the project path

        If ``package.__file__`` is ``/foo/foo/__init__.py``, then project.path
        should be ``/foo``.
        """
        return pathlib.Path(self.package.__file__).resolve().parent.parent

    @property
    def repo(self):
        """Return a git.Repo object corresponding to this project"""
        return git.Repo(self.path, search_parent_directories=True)

    @property
    def contrib_module_path(self):
        """Return the path to the project's contrib module"""
        return self.get('contrib', 'module_path')

    def __getattr__(self, attr):
        if attr in Project.attr_map:
            return self._resolve(*Project.attr_map[attr])
        else:
            return object.__getattribute__(self, attr)
