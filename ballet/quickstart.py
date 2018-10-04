from cookiecutter.main import cookiecutter

from ballet.compat import pathlib

def generate_project(**kwargs):
    path = pathlib.Path(__file__).resolve().parent.joinpath('project_template')
    path = str(path)
    cookiecutter(path, **kwargs)

def main():
    generate_project()
