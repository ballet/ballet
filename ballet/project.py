import inspect
import pathlib
import sys
from functools import partial
from importlib import import_module
from types import ModuleType
from typing import Any, Callable, List, Optional, Tuple

import git
from dynaconf import Dynaconf
from funcy import cache, cached_property, fallback, re_find
from pandas import DataFrame

import ballet.contrib
from ballet.compat import is_mount
from ballet.eng import BaseTransformer
from ballet.exc import ConfigurationError
from ballet.feature import Feature
from ballet.pipeline import (
    EngineerFeaturesResult, FeatureEngineeringPipeline,
    make_engineer_features,)
from ballet.util import raiseifnone
from ballet.util.ci import get_travis_branch
from ballet.util.git import get_branch
from ballet.util.mod import import_module_at_path
from ballet.util.typing import Pathy

DEFAULT_CONFIG_NAME = 'ballet.yml'
DYNACONF_OPTIONS = {
    'envvar_prefix': 'BALLET',
    'settings_file': DEFAULT_CONFIG_NAME,
    'yaml_loader': 'safe_load',
}


def load_config(path: Optional[Pathy] = None, ascend: bool = True) -> Dynaconf:
    """User-facing function to load config from project code

    The default behavior when no arguments are provided is to detect the
    calling code using introspection and load a config object by ascending
    the directory of the calling code. If this does not succeed, you should
    just pass `path` directly.
    """
    if path is None:
        # "The first entry in the returned list represents the caller; the
        # last entry represents the outermost call on the stack."
        frame = inspect.stack()[1]
        try:
            path = frame.filename
        finally:
            del frame

    path = pathlib.Path(path)
    while path.exists() and not is_mount(path):
        try:
            return load_config_in_dir(path)
        except ConfigurationError:
            if ascend:
                path = path.parent

    raise ConfigurationError


def load_config_at_path(path: Pathy) -> Dynaconf:
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
            'root_path': str(path.parent),
            'settings_file': str(path.name),
        })
        return Dynaconf(**options)
    else:
        raise ConfigurationError(
            f'Couldn\'t find ballet.yml config file at {path!s}')


def load_config_in_dir(path: Pathy) -> Dynaconf:
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
        f'user_{username}', f'feature_{featurename}.py')


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
        # protocol:user/repo, i.e. 'git@github.com:ballet/ballet'
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
    def config(self) -> Dynaconf:
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
    def branch(self) -> Optional[str]:
        """Return current git branch according to git tree or CI environment"""
        result = get_branch(repo=self.repo)
        if result is None:
            result = get_travis_branch()
        return result

    @property
    def on_master(self) -> bool:
        return self.branch == 'master'

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

    @property
    def version(self) -> str:
        """Some version identifier for the current project

        Implementation is to return the abbreviated SHA1 of git HEAD.
        """
        return self.repo.head.commit.hexsha[:7]


class FeatureEngineeringProject:

    CACHE_TIMEOUT = 10 * 60

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
        self._load_data = cache(self.CACHE_TIMEOUT)(load_data)
        self._extra_features = extra_features or []
        self._engineer_features = engineer_features

        # use fqn to avoid circular import
        self.collect = partial(ballet.contrib.collect_contrib_features,
                               self.project)

    @cached_property
    def project(self) -> Project:
        """Get the Project object representing this project."""
        return Project(self._package)

    @property
    def features(self) -> List[Feature]:
        """Get all features from the project

        Both collects all contrib features from the project and allows extra
        features to be provided by the API author.
        """
        return self.collect() + self._extra_features

    @property
    def pipeline(self) -> FeatureEngineeringPipeline:
        """Get the feature engineering pipeline from the existing features"""
        return FeatureEngineeringPipeline(self.features)

    def load_data(self, *args, cache=True, **kwargs):
        """Call the project's load_data function, caching dataset

        Dataset is cached for `FeatureEngineeringProject.CACHE_TIMEOUT`
        seconds. To invalidate cache and cause data to be re-loaded from
        wherever it comes from, pass `cache=False`.
        """
        if not cache:
            self._load_data.invalidate_all()
        return self._load_data(*args, **kwargs)

    def engineer_features(self, *args, **kwargs) -> EngineerFeaturesResult:
        """Engineer features"""
        if self._engineer_features is None:
            self._engineer_features = make_engineer_features(
                self.pipeline, self.encoder, self.load_data)

        return self._engineer_features(*args, **kwargs)
