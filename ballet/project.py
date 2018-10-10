from importlib import import_module

import git
import yaml
from funcy import cached_property, get_in, memoize, partial

from ballet.compat import pathlib
from ballet.exc import ConfigurationError, Error
from ballet.util.ci import get_travis_pr_num
from ballet.util.git import get_pr_num

DEFAULT_CONFIG_NAME = 'ballet.yml'


def get_config_paths(package_root):
    """Get candidate config paths

    Creates a sequence of paths that includes the package root and all of its
    parents, as well as ~/.ballet.
    """
    package_root = pathlib.Path(package_root)

    # parents of package directory
    paths = [
        d.joinpath(DEFAULT_CONFIG_NAME)
        for d in package_root.parents
    ]

    # home directory
    paths.append(
        pathlib.Path.home().joinpath('.ballet', DEFAULT_CONFIG_NAME))

    # defaults in ballet repo

    return paths


@memoize
def find_configs(package_root):
    """Find valid ballet project config files

    See if any of the candidates returned by get_config_paths are valid.

    Raises:
        ConfigurationError: No valid config files were found.
    """
    configs = []
    for candidate in get_config_paths(package_root):
        if candidate.exists() and candidate.is_file():
            with candidate.open('r') as f:
                config = yaml.load(f)
                configs.append(config)

    if configs:
        return configs
    else:
        raise ConfigurationError("Couldn't find any ballet.yml config files.")


def config_get(package_root, *path, default=None):
    configs = find_configs(package_root)

    o = object()
    for config in configs:
        result = get_in(config, path, default=o)
        if result is not o:
            return result

    return default


def make_config_get(package_root):
    return partial(config_get, package_root)


class Project:

    attr_map = {
        'conf': ('.conf', None),
        'get': ('.conf', 'get'),
        'load_data': ('.load_data', 'load_data'),
        'build_features': ('.features.build_features', 'build_features'),
        'get_contrib_features': ('.features.build_features',
                                 'get_contrib_features')
    }

    def __init__(self, package):
        self.package = package

    def _resolve(self, modname, attr=None):
        module = import_module(modname, package=self.package.__name__)
        if attr is not None:
            return getattr(module, attr)
        else:
            return module

    @cached_property
    def pr_num(self):
        result = get_pr_num(repo=self.repo)
        if result is None:
            result = get_travis_pr_num()
        if result is None:
            raise Error

    @cached_property
    def path(self):
        return pathlib.Path(self.package.__file__).resolve()

    @cached_property
    def repo(self):
        return git.Repo(self._path, search_parent_directories=True)

    def __getattr__(self, attr):
        if attr in Project.attr_map:
            return self._resolve(*Project.attr_map[attr])
        else:
            raise AttributeError
