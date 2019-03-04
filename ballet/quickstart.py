from cookiecutter.main import cookiecutter

from ballet.compat import pathlib, safepath

PROJECT_TEMPLATE_PATH = (
    pathlib.Path(__file__).resolve().parent.joinpath('project_template'))


def _get_project_template_path():
    return safepath(PROJECT_TEMPLATE_PATH)


def generate_project(**kwargs):
    """Generate a ballet project according to the project template

    Args:
        **kwargs: options for the cookiecutter template
    """
    # must use str because cookiecutter can't handle path-like objects.
    project_template_path = str(_get_project_template_path())
    if 'output_dir' in kwargs:
        kwargs['output_dir'] = str(kwargs['output_dir'])
    return cookiecutter(project_template_path, **kwargs)


def main():
    """Entry point for ballet-quickstart command line tool"""
    return generate_project()
