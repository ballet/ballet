#!/usr/bin/env python

import git

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


def main():
    create_git_repo()


if __name__ == '__main__':
    main()
