import pathlib
import sys
from importlib import import_module

import git
from dynaconf import LazySettings
from funcy import cached_property, fallback, re_find

from ballet.compat import safepath
from ballet.exc import ConfigurationError
from ballet.util import needs_path, raiseifnone
from ballet.util.ci import get_travis_branch, get_travis_pr_num
from ballet.util.git import get_branch, get_pr_num, is_merge_commit
from ballet.util.mod import import_module_at_path

DEFAULT_CONFIG_NAME = 'ballet.yml'
DYNACONF_OPTIONS = {
    'ENVVAR_PREFIX_FOR_DYNACONF': 'BALLET',
    'SETTINGS_FILE_FOR_DYNACONF': DEFAULT_CONFIG_NAME,
    'YAML_LOADER': 'safe_load',
}


config = LazySettings(**DYNACONF_OPTIONS)


@needs_path
def load_config_at_path(path):
    """Load config at exact path

    Args:
        path (path-like): path to config file

    Returns:
        dict: config dict
    """
    if path.exists() and path.is_file():
        options = DYNACONF_OPTIONS.copy()
        options.update({
            'ROOT_PATH_FOR_DYNACONF': safepath(path.parent),
            'SETTINGS_FILE_FOR_DYNACONF': safepath(path.name),
        })
        return LazySettings(**options)
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
    return load_config_at_path(path.joinpath(DEFAULT_CONFIG_NAME))


def relative_to_contrib(diff, project):
    """Compute relative path of changed file to contrib dir

    Args:
        diff (git.diff.Diff): file diff
        project (Project): project

    Returns:
        Path
    """
    path = pathlib.Path(diff.b_path)
    contrib_path = project.config.get('contrib.module_path')
    return path.relative_to(contrib_path)


@needs_path
def make_feature_path(contrib_dir, username, featurename):
    return contrib_dir.joinpath(
        'user_{}'.format(username), 'feature_{}.py'.format(featurename))


def detect_github_username(project):
    """Detect github username

    Looks in the following order:
    1. github.user git config variable
    2. git remote origin
    3. $USER
    4. 'username'
    """
    @raiseifnone
    def get_config_variable():
        return project.repo.config_reader().get_value(
            'github', 'user', default=None)

    @raiseifnone
    def get_remote():
        url = list(project.repo.remote('origin').urls)[0]
        # protocol:user/repo, i.e. 'git@github.com:HDI-Project/ballet'
        return re_find(r'.+:(.+)/.+', url)

    @raiseifnone
    def get_user_env():
        return sys.environ.get('USER')

    def get_default():
        return 'username'

    return fallback(
        get_config_variable, get_remote, get_user_env, (get_default, ())
    )


class Project:
    """Encapsulate information on a ballet project

    This is a utility class mostly useful for easy access to the project's
    information from within the ballet.validation package.

    In addition to the defined methods and properties, the following functions
    of the project can be accessed as attributes of a class instance, where
    ``prj`` refers to the python module of the underlying ballet project:
    - ``load_data`` (``prj.load_data.load_data``)
    - ``build`` (``prj.features.build``)
    - ``collect_contrib_features`` (``prj.features.collect_contrib_features``)

    Args:
        package (ModuleType): python package representing imported ballet
            project
    """

    attr_map = {
        'load_data': ('.load_data', 'load_data'),
        'build': ('.features', 'build'),
        'collect_contrib_features': ('.features', 'collect_contrib_features')
    }

    def __init__(self, package):
        self.package = package

    @cached_property
    def config(self):
        return load_config_in_dir(self.path)

    @classmethod
    def from_path(cls, path):
        """Create a Project instance from an fs path to the containing dir

        Args:
            path (PathLike): path to directory that contains the
                project
        """
        path = pathlib.Path(path)
        config = load_config_in_dir(path)
        package_slug = config.get('project.package_slug')
        package = import_module_at_path(package_slug,
                                        path.joinpath('src', package_slug))
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

    @property
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

    @property
    def on_master(self):
        return self.branch == 'master'

    @property
    def on_master_after_merge(self):
        """Check the repo HEAD is on master after a merge commit

        Checks for two qualities of the current project:
        1. The project repo's head is the master branch
        2. The project repo's head commit is a merge commit.

        Note that fast-forward style merges will not cause the second condition
        to evaluate to true.
        """

        return self.on_master and is_merge_commit(self.repo.head.commit)

    @property
    def path(self):
        """Return the project path (aka project root)

        If ``package.__file__`` is ``/foo/src/foo/__init__.py``,
        then project.path
        should be ``/foo``.
        """
        return pathlib.Path(self.package.__file__).resolve().parents[2]

    @property
    def repo(self):
        """Return a git.Repo object corresponding to this project"""
        return git.Repo(self.path, search_parent_directories=True)

    def __getattr__(self, attr):
        if attr in Project.attr_map:
            return self._resolve(*Project.attr_map[attr])
        else:
            return object.__getattribute__(self, attr)
