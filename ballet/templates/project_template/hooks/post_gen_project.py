#!/usr/bin/env python

import collections
import json
import pathlib

import git

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
    repo.index.commit('Automatically generated files from ballet quickstart')
    repo.create_remote('origin',
                       'https://github.com/{{ cookiecutter.github_owner }}'
                       '/{{ cookiecutter.project_slug }}')
    repo.create_head(TEMPLATE_BRANCH)


def clean_cookiecutter_context():
    fn = '.cookiecutter_context.json'
    with open(fn, 'r') as f:
        context = json.load(f, object_pairs_hook=collections.OrderedDict)

    # strip _template key
    context['cookiecutter'].pop('_template')

    with open(fn, 'w') as f:
        json.dump(context, f)


def echo():
    fn = pathlib.Path.cwd().resolve()
    logger.info(f'New project created in {fn!s}')


def main():
    clean_cookiecutter_context()
    create_git_repo()
    echo()


if __name__ == '__main__':
    main()
