import yaml
from funcy import get_in, memoize, partial

from ballet.compat import pathlib
from ballet.exc import ConfigurationError

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
