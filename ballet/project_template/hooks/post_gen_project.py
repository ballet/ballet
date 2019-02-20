#!/usr/bin/env python

import git
import os
import shutil

PROJECT_DIRECTORY = os.path.realpath(os.path.curdir)

def copy_context():
    src = os.path.join(os.path.expanduser('~'),
                       '.cookiecutter_replay',
                       'project_template.json')
    dst = os.path.join(PROJECT_DIRECTORY, '.cookiecutter_replay.json')
    shutil.copyfile(src, dst)

def create_git_repo():
    repo = git.Repo.init('.')
    with repo.config_writer() as writer:
        writer.set_value('user', 'name', '{{ cookiecutter.full_name }}')
        writer.set_value('user', 'email', '{{ cookiecutter.email }}')
        writer.set_value('github', 'user', '{{ cookiecutter.github_owner }}')
        writer.release()
    repo.git.add('-u')
    repo.index.commit('Automatically generated files from ballet-quickstart')
    repo.create_remote('origin', 'https://github.com/{{ cookiecutter.github_owner }}/{{ cookiecutter.project_slug }}')



def main():
    copy_context()
    create_git_repo()


if __name__ == '__main__':
    main()