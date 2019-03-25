from cookiecutter.main import cookiecutter as _cookiecutter
from funcy import re_test, walk, walk_values, wraps

from ballet.compat import pathlib, safepath
from ballet.util.fs import ispathlike, synctree

TEMPLATES_PATH = pathlib.Path(__file__).resolve().parent.joinpath('templates')
FEATURE_TEMPLATE_PATH = TEMPLATES_PATH.joinpath('feature_template')
PROJECT_TEMPLATE_PATH = TEMPLATES_PATH.joinpath('project_template')


def _stringify_path(obj):
    return str(obj) if ispathlike(obj) else obj


@wraps(_cookiecutter)
def cookiecutter(*args, **kwargs):
    args = walk(_stringify_path, args)
    kwargs = walk_values(_stringify_path, kwargs)
    return _cookiecutter(*args, **kwargs)


def render_project_template(**cc_kwargs):
    """Generate a ballet project according to the project template

    Args:
        **cc_kwargs: options for the cookiecutter template
    """
    project_template_path = PROJECT_TEMPLATE_PATH
    return cookiecutter(project_template_path, **cc_kwargs)


def render_feature_template(**cc_kwargs):
    """Create a stub for a new feature

    Args:
        **cc_kwargs: options for the cookiecutter template
    """
    feature_template_path = FEATURE_TEMPLATE_PATH
    return cookiecutter(feature_template_path, **cc_kwargs)


    """
    # must use str because cookiecutter can't handle path-like objects.
    feature_template_path = str(FEATURE_TEMPLATE_PATH)
    if 'output_dir' in kwargs:
        kwargs['output_dir'] = str(kwargs['output_dir'])
    return cookiecutter(feature_template_path, **kwargs)
