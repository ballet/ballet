import pathlib
import sys
from functools import partial
from importlib import import_module
from types import ModuleType
from typing import Any, Callable, List, Tuple

import git
from dynaconf import LazySettings
from funcy import cached_property, fallback, re_find
from pandas import DataFrame

import ballet.contrib
from ballet.compat import is_mount
from ballet.eng import BaseTransformer
from ballet.exc import ConfigurationError
from ballet.feature import Feature
from ballet.pipeline import (
    EngineerFeaturesResult, FeatureEngineeringPipeline, make_engineer_features)
from ballet.util import raiseifnone
from ballet.util.ci import get_travis_branch, get_travis_pr_num
from ballet.util.git import get_branch, get_pr_num, is_merge_commit
from ballet.util.mod import import_module_at_path
from ballet.util.typing import Pathy

DEFAULT_CONFIG_NAME = 'ballet.yml'
DYNACONF_OPTIONS = {
    'ENVVAR_PREFIX_FOR_DYNACONF': 'BALLET',
    'SETTINGS_FILE_FOR_DYNACONF': DEFAULT_CONFIG_NAME,
    'YAML_LOADER': 'safe_load',
}


config = LazySettings(**DYNACONF_OPTIONS)


def load_config_at_path(path: Pathy) -> LazySettings:
    """Load config at exact path

    Args:
        path: path to config file

    Returns:
        dict: config dict
    """
    path = pathlib.Path(path)
    if path.exists() and path.is_file():
        options = DYNACONF_OPTIONS.copy()
        options.update({
            'ROOT_PATH_FOR_DYNACONF': str(path.parent),
            'SETTINGS_FILE_FOR_DYNACONF': str(path.name),
        })
        return LazySettings(**options)
    else:
        raise ConfigurationError(
            'Couldn\'t find ballet.yml config file at {path!s}'
            .format(path=path))


def load_config_in_dir(path: Pathy) -> LazySettings:
    """Load config in containing directory

    Args:
        path: path to containing directory of config file

    Returns:
        config dict
    """
    path = pathlib.Path(path)
    return load_config_at_path(path.joinpath(DEFAULT_CONFIG_NAME))


def relative_to_contrib(
    diff: git.diff.Diff, project: 'Project'
) -> pathlib.Path:
    """Compute relative path of changed file to contrib dir

    Args:
        diff: file diff
        project: project
    """
    path = pathlib.Path(diff.b_path)
    contrib_path = project.config.get('contrib.module_path')
    return path.relative_to(contrib_path)


def make_feature_path(
    contrib_dir: Pathy, username: str, featurename: str
) -> pathlib.Path:
    contrib_dir = pathlib.Path(contrib_dir)
    return contrib_dir.joinpath(
        'user_{}'.format(username), 'feature_{}.py'.format(featurename))


def detect_github_username(project: 'Project') -> str:
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

    Args:
        package (ModuleType): python package representing imported ballet
            project
    """

    def __init__(self, package: ModuleType):
        self.package = package

    @cached_property
    def config(self) -> LazySettings:
        return load_config_in_dir(self.path)

    @classmethod
    def from_path(cls, path: Pathy, ascend: bool = False):
        """Create a Project instance from an fs path to the containing dir

        Args:
            path: path to directory that contains the project
            ascend: if the config file is not found in the given directory,
                then search in parent directories, stopping at a file system
                boundary
        """
        path = pathlib.Path(path)
        try:
            config = load_config_in_dir(path)
            package_slug = config.get('project.package_slug')
            package = import_module_at_path(package_slug,
                                            path.joinpath('src', package_slug))
            return cls(package)
        except ConfigurationError:
            if ascend:
                parent = path.parent
                if parent.exists() and not is_mount(parent):
                    return cls.from_path(parent, ascend=ascend)
            raise

    @classmethod
    def from_cwd(cls):
        """Create a Project instance by searching up from cwd

        Recursively searches for the ballet configuration file at the
        current working directory and parent directories, stopping when it
        reaches a file system boundary.

        Raises:
            ConfigurationError: couldn't find the configuration file
        """
        cwd = pathlib.Path.cwd()
        return cls.from_path(cwd, ascend=True)

    def resolve(
        self, modname: str, attr: str = None
    ) -> Any:
        """Import module or attribute from project

        Args:
            modname: dotted module name relative to top-level with leading
                dot omited; if trying to import the top-level package,
                use '' (can also just access self.package)
            attr: attribute to get from the imported module

        Example:

            >>> project.resolve('', '__version__')
            # return __version__ attribute from top-level package
            >>> project.resolve('api')
            # return myproject.api module
            >>> project.resolve('api', attr='api')
            # return api object from myproject.api module
            >>> project.resolve('foo.bar')
            # return myproject.foo.bar module
        """

        if modname:
            module = import_module('.' + modname,
                                   package=self.package.__name__)
        else:
            module = self.package

        if attr is not None:
            return getattr(module, attr)
        else:
            return module

    @property
    def pr_num(self) -> str:
        """Return the PR number or None if not on a PR"""
        result = get_pr_num(repo=self.repo)
        if result is None:
            result = get_travis_pr_num()
        return result

    @property
    def on_pr(self) -> bool:
        """Return whether the project has a source tree on a PR"""
        return self.pr_num is not None

    @property
    def branch(self) -> str:
        """Return current git branch according to git tree or CI environment"""
        result = get_branch(repo=self.repo)
        if result is None:
            result = get_travis_branch()
        return result

    @property
    def on_master(self) -> bool:
        return self.branch == 'master'

    @property
    def on_master_after_merge(self) -> bool:
        """Check the repo HEAD is on master after a merge commit

        Checks for two qualities of the current project:
        1. The project repo's head is the master branch
        2. The project repo's head commit is a merge commit.

        Note that fast-forward style merges will not cause the second condition
        to evaluate to true.
        """

        return self.on_master and is_merge_commit(self.repo.head.commit)

    @cached_property
    def path(self) -> pathlib.Path:
        """Return the project path (aka project root)

        If ``package.__file__`` is ``/foo/src/foo/__init__.py``,
        then project.path
        should be ``/foo``.
        """
        return pathlib.Path(self.package.__file__).resolve().parents[2]

    @cached_property
    def repo(self) -> git.Repo:
        """Return a git.Repo object corresponding to this project"""
        return git.Repo(self.path, search_parent_directories=True)

    @property
    def api(self) -> 'FeatureEngineeringProject':
        return self.resolve('api', 'api')


class FeatureEngineeringProject:

    def __init__(
        self,
        *,
        package: ModuleType,
        encoder: BaseTransformer,
        load_data: Callable[..., Tuple[DataFrame, DataFrame]],
        extra_features: List[Feature] = None,
        engineer_features: Callable[..., EngineerFeaturesResult] = None,
    ):
        self._package = package
        self.encoder = encoder
        self.load_data = load_data
        self._extra_features = extra_features or []
        self._engineer_features = engineer_features

        # use fqn to avoid circular import
        self.collect = partial(ballet.contrib.collect_contrib_features,
                               self.project)

    @cached_property
    def project(self) -> Project:
        return Project(self._package)

    @property
    def features(self) -> List[Feature]:
        return self.collect() + self._extra_features

    @property
    def pipeline(self) -> FeatureEngineeringPipeline:
        return FeatureEngineeringPipeline(self.features)

    def engineer_features(self, *args, **kwargs) -> EngineerFeaturesResult:
        """Engineer features"""
        if self._engineer_features is None:
            self._engineer_features = make_engineer_features(
                self.pipeline, self.encoder, self.load_data)

        return self._engineer_features(*args, **kwargs)
