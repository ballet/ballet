from cookiecutter.main import cookiecutter

from ballet.compat import pathlib

PROJECT_TEMPLATE_PATH = (
    pathlib.Path(__file__).resolve().parent.joinpath('project_template'))


def generate_project(**kwargs):
    """Generate a ballet project according to the project template

    Args:
        **kwargs: options for the cookiecutter template
    """
    cookiecutter(str(PROJECT_TEMPLATE_PATH), **kwargs)


def main():
    """Entry point for ballet-quickstart command line tool"""
    generate_project()
