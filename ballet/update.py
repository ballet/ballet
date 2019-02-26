import json
import os
import tempfile

import funcy
import git
from cookiecutter.prompt import prompt_for_config

from ballet import __version__ as version
from ballet.compat import pathlib, safepath
from ballet.quickstart import generate_project
from ballet.util.log import logger

PROJECT_CONTEXT_PATH = (
    pathlib.Path(__file__).resolve().parent.joinpath(
        'project_template',
        'cookiecutter.json'))

REPLAY_PATH = (
    pathlib.Path.home().joinpath(
        '.cookiecutter_replay',
        'project_template.json'))

TEMPLATE_BRANCH = 'template-update'


def _make_master_branch_merge_commit_message():
    return 'Merge project template updates from ballet v{}'.format(version)


def _create_replay(cwd, tempdir):
    tempdir = pathlib.Path(tempdir)
    context = _get_full_context(cwd)
    slug = context['cookiecutter']['project_slug']
    old_context = None
    try:
        if REPLAY_PATH.exists():
            with open(REPLAY_PATH, 'r') as replay_file:
                old_context = json.load(replay_file)
        # if there are old replays, save it before it's overwritten
        with open(REPLAY_PATH, 'w') as replay_file:
            json.dump(context, replay_file)
        # load our context and prompt as necessary
        generate_project(
            replay=True,
            output_dir=safepath(tempdir))
    except BaseException:
        # we're missing keys, figure out which and prompt
        logger.exception(
            'Could not create a updated project copy, update failed.')
    finally:
        # put back the old replay, if it's been overwritten
        if old_context is not None:
            with open(REPLAY_PATH, 'w') as replay_file:
                json.dump(old_context, replay_file)
    return tempdir / slug


def _get_full_context(cwd):
    context_path = cwd.joinpath('.cookiecutter_replay.json')
    if context_path.exists():
        with open(context_path) as context_file:
            context = json.load(context_file)
    else:
        raise FileNotFoundError(
            'Could not find cookiecutter_replay.json, '
            'are you in a ballet repo?')
    with open(PROJECT_CONTEXT_PATH) as context_json:
        new_context = json.load(context_json)
    new_keys = set(new_context.keys()) - set(context['cookiecutter'].keys())
    new_context_config = {'cookiecutter': funcy.project(new_context, new_keys)}
    new_context = prompt_for_config(new_context_config)
    context['cookiecutter'].update(new_context)
    return context


def update_project_template(create_merge_commit=False):
    cwd = pathlib.Path(os.getcwd())
    try:
        current_repo = git.Repo(
            safepath(cwd),
            search_parent_directories=True)  # for right now
    except BaseException:
        msg = 'Could not find ballet repo, update failed'
        logger.exception(msg)

    with tempfile.TemporaryDirectory() as tempdir:
        updated_template = _create_replay(cwd, tempdir)
        updated_repo = git.Repo(safepath(updated_template))
        # add some randomness in the remote name by using tempdir
        remote_name = updated_template.parts[-1]
        try:
            current_repo.heads[TEMPLATE_BRANCH].checkout()
            updated_remote = current_repo.create_remote(
                remote_name, updated_repo.working_tree_dir)
            updated_remote.fetch()
            current_repo.git.merge(
                remote_name + '/master',
                allow_unrelated_histories=True,
                strategy_option='theirs',
                squash=True,
            )
            current_repo.index.commit(
                _make_master_branch_merge_commit_message())
        except BaseException:
            msg = ('Could not merge changes into template-update branch, '
                   'update failed')
            logger.exception(msg)
        finally:
            if current_repo.active_branch != current_repo.heads.master:
                current_repo.heads.master.checkout()
            current_repo.delete_remote(remote_name)

    try:
        current_repo.git.merge(
            TEMPLATE_BRANCH,
            squash=True,
        )
        if not create_merge_commit:
            commit_prompt = ('Would you like ballet to create a merge commit '
                             'automatically? [y/N]: ')
            answer = input(commit_prompt)
            if 'y' in answer.lower():
                create_merge_commit = True
        if create_merge_commit:
            current_repo.index.commit(
                _make_master_branch_merge_commit_message())
    except BaseException:
        msg = 'Could not merge changes into master, update failed'
        logger.exception(msg)


def main():
    update_project_template()
