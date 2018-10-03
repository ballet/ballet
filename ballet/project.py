import yaml
from ballet.compat import pathlib
from ballet.exc import ConfigurationError
from funcy import get_in, partial


def find_ballet_yml(package_root):
    package_root = pathlib.Path(package_root)
    dirs = list(package_root.parents) + [pathlib.Path.home().joinpath('.ballet')]
    for d in dirs:
        candidate = d.joinpath('ballet.yml')
        if candidate.exists() and candidate.is_file():
            with candidate.open('r') as f:
                return yaml.load(f)

    raise ConfigurationError


def config_get(package_root, *path, default=None):
    config = find_ballet_yml(package_root)
    return get_in(config, path, default=default)


def make_config_get(package_root):
    return partial(config_get, package_root)
