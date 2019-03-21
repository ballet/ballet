from cookiecutter.main import cookiecutter

from ballet.compat import pathlib, safepath

TEMPLATES_PATH = pathlib.Path(__file__).resolve().parent.joinpath('templates')
FEATURE_TEMPLATE_PATH = TEMPLATES_PATH.joinpath('feature_template')
PROJECT_TEMPLATE_PATH = TEMPLATES_PATH.joinpath('project_template')


def _get_project_template_path():
    return safepath(PROJECT_TEMPLATE_PATH)



def render_project_template(**kwargs):
    """Generate a ballet project according to the project template

    Args:
        **kwargs: options for the cookiecutter template
    """
    # must use str because cookiecutter can't handle path-like objects.
    project_template_path = str(_get_project_template_path())
    if 'output_dir' in kwargs:
        kwargs['output_dir'] = str(kwargs['output_dir'])
    return cookiecutter(project_template_path, **kwargs)


def render_feature_template(**kwargs):
    """Create a stub for a new feature

    Args:
        **kwargs: options for the cookiecutter template
    """
    # must use str because cookiecutter can't handle path-like objects.
    feature_template_path = str(FEATURE_TEMPLATE_PATH)
    if 'output_dir' in kwargs:
        kwargs['output_dir'] = str(kwargs['output_dir'])
    return cookiecutter(feature_template_path, **kwargs)
