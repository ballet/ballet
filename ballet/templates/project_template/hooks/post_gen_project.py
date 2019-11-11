#!/usr/bin/env python

import git
import json
import pathlib

from ballet.util.log import logger
from ballet.update import TEMPLATE_BRANCH


def create_git_repo():
    repo = git.Repo.init('.')
    with repo.config_writer() as writer:
        writer.set_value('user', 'name', '{{ cookiecutter.full_name }}')
        writer.set_value('user', 'email', '{{ cookiecutter.email }}')
        writer.set_value('github', 'user', '{{ cookiecutter.github_owner }}')
        writer.release()
    repo.git.add('.')
    repo.index.commit('Automatically generated files from ballet-quickstart')
    repo.create_remote('origin',
                       'https://github.com/{{ cookiecutter.github_owner }}'
                       '/{{ cookiecutter.project_slug }}')
    repo.create_head(TEMPLATE_BRANCH)


def clean_cookiecutter_context():
    fn = pathlib.Path('.cookiecutter_context.json')
    with open(fn, 'r') as f:
        j = json.load(f)

    # strip _template key
    j['cookiecutter'].pop('_template')

    with open(fn, 'w') as f:
        json.dump(j, f)


def echo():
    fn = pathlib.Path.cwd().absolute()
    logger.info('New project created in {}'.format(fn))


def main():
    clean_cookiecutter_context()
    create_git_repo()
    echo()


if __name__ == '__main__':
    main()
