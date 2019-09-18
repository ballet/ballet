import pathlib
from importlib import import_module

import git
import yaml
from funcy import get_in, partial

from ballet.exc import ConfigurationError
from ballet.util import needs_path
from ballet.util.ci import get_travis_branch, get_travis_pr_num
from ballet.util.git import get_branch, get_pr_num, is_merge_commit
from ballet.util.mod import import_module_at_path

DEFAULT_CONFIG_NAME = 'ballet.yml'


@needs_path
def get_config_path(project_root):
    """Get candidate config path

    Args:
        project_root (path-like): Directory of the ballet project root
            directory, the one usually containing the ``ballet.yml`` file.
    """
    return project_root.resolve().joinpath(DEFAULT_CONFIG_NAME)


@needs_path
def load_config_at_path(path):
    """Load config at exact path

    Args:
        path (path-like): path to config file

    Returns:
        dict: config dict
    """
    if path.exists() and path.is_file():
        with path.open('r') as f:
            return yaml.load(f, Loader=yaml.SafeLoader)
    else:
        raise ConfigurationError("Couldn't find ballet.yml config file.")


@needs_path
def load_config_in_dir(path):
    """Load config in containing directory

    Args:
        path (path-like): path to containing directory of config file

    Returns:
        dict: config dict
    """
    candidate = get_config_path(path)
    return load_config_at_path(candidate)


def config_get(config, *path, default=None):
    """Get a configuration option following a path through the config

    Example usage:

        >>> config_get(config,
                       'problem', 'problem_type_details', 'scorer',
                       default='accuracy')

    Args:
        config (dict): config dict
        *path (list[str]): List of config sections and options to follow.
        default (default=None): A default value to return in the case that
            the option does not exist.
    """
    o = object()
    result = get_in(config, path, default=o)
    if result is not o:
        return result
    else:
        return default


@needs_path
def _get_project_root_from_conf_path(conf_path):
    if conf_path.name == 'conf.py':
        return conf_path.parent.parent.resolve()
    else:
        raise ValueError("Must pass path to conf module")


@needs_path
def make_config_get(conf_path):
    """Return a function to get configuration options for a specific project

    Args:
        conf_path (path-like): path to project's conf file (i.e. foo.conf
            module)
    """
    project_root = _get_project_root_from_conf_path(conf_path)
    config = load_config_in_dir(project_root)
    return partial(config_get, config)


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
    - ``collect_contrib_features`` (``prj.features.collect_contrib_features``)

    Args:
        package (ModuleType): python package representing imported ballet
            project
    """

    attr_map = {
        'conf': ('.conf', None),
        'get': ('.conf', 'get'),
        'load_data': ('.load_data', 'load_data'),
        'build': ('.features', 'build'),
        'collect_contrib_features': ('.features', 'collect_contrib_features')
    }

    def __init__(self, package):
        self.package = package

    @classmethod
    def from_path(cls, path):
        config = load_config_in_dir(path)
        project_slug = config_get(config, 'project', 'slug')
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
    def branch(self):
        """Return whether the project is on master branch"""
        result = get_branch(repo=self.repo)
        if result is None:
            result = get_travis_branch()
        return result

    def on_master(self):
        return self.branch == 'master'

    def on_master_after_merge(self):
        """Checks for two qualities of the current project:
        1. The project repo's head is the master branch
        2. The project repo's head commit is a merge commit.
        """

        return self.on_master() and is_merge_commit(self.repo.head.commit)

    @property
    def path(self):
        """Return the project path (aka project root)

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
