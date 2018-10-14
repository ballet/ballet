from cookiecutter.main import cookiecutter

from ballet.compat import pathlib

PROJECT_TEMPLATE_PATH = (
    pathlib.Path(__file__).resolve().parent.joinpath('project_template'))


def generate_project(**kwargs):
    cookiecutter(str(PROJECT_TEMPLATE_PATH), **kwargs)


def main():
    generate_project()
