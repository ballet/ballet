import yaml
from ballet.compat import pathlib
from ballet.exc import ConfigurationError
from funcy import get_in, merge, partial


def find_ballet_yml(package_root):
    """Find ballet.yml config file

    Recursively search up the directory tree from the package root, and then also search in
    """
    package_root = pathlib.Path(package_root)
    dirs = list(package_root.parents) + [pathlib.Path.home().joinpath('.ballet')]
    configs = []
    for d in dirs:
        candidate = d.joinpath('ballet.yml')
        if candidate.exists() and candidate.is_file():
            with candidate.open('r') as f:
                config = yaml.load(f)
                configs.append(config)

    if configs:
        return configs
    else:
        raise ConfigurationError("Couldn't find any ballet.yml config files.")


def config_get(package_root, *path, default=None):
    configs = find_ballet_yml(package_root)
    config = merge(*configs)
    return get_in(config, path, default=default)


def make_config_get(package_root):
    return partial(config_get, package_root)
