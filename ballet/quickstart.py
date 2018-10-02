from cookiecutter.main import cookiecutter

from ballet.compat import pathlib

def generate_project():
    path = pathlib.Path.cwd().parent.joinpath('project_template')
    path = str(path)
    cookiecutter(path)

def main():
    generate_project()
