from cookiecutter.main import cookiecutter

from ballet.compat import pathlib

PROJECT_TEMPLATE_PATH = (
    pathlib.Path(__file__).resolve().parent.joinpath('project_template'))


def _get_project_template_path():
    return str(PROJECT_TEMPLATE_PATH)


def generate_project(**kwargs):
    """Generate a ballet project according to the project template

    Args:
        **kwargs: options for the cookiecutter template
    """
    project_template_path = _get_project_template_path()
    return cookiecutter(project_template_path, **kwargs)


def main():
    """Entry point for ballet-quickstart command line tool"""
    return generate_project()
