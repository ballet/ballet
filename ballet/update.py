import funcy
import git
import json
import os
import tempfile
import yaml

from cookiecutter.main import cookiecutter
from cookiecutter.prompt import prompt_for_config

from ballet.compat import pathlib, safepath
from ballet.util.log import logger
from ballet.project import get_config_paths, find_configs
from ballet.quickstart import generate_project

PROJECT_CONTEXT_PATH = (
    pathlib.Path(__file__).resolve().parent.joinpath('project_template', 'cookiecutter.json')
)

REPLAY_PATH = (
    pathlib.Path.home().joinpath('.cookiecutter_replay', 'project_template.json')
)

def _create_replay(tempdir, context):
    tempdir = pathlib.Path(tempdir)
    name = context['cookiecutter']['project_name']
    old_context = None
    try:
        # replace the current replay with our own
        with open(REPLAY_PATH, 'r') as replay_file:
            old_context = json.load(replay_file)
        with open(REPLAY_PATH, 'w') as replay_file:
            json.dump(context, replay_file)
        generate_project(replay=True, output_dir=tempdir)
    except:
        # we're missing keys, figure out which and prompt
        with open(PROJECT_CONTEXT_PATH) as context_json:
            new_context = json.load(context_json)
        new_keys = set(new_context.keys()) - set(context['cookiecutter'].keys())
        new_context_config = {'cookiecutter': funcy.project(new_context, new_keys)}
        new_context = prompt_for_config(new_context_config)
        context['cookiecutter'].update(new_context)

        with open(REPLAY_PATH, 'w') as replay_file:
            json.dump(context, replay_file)
        generate_project(replay=True, output_dir=tempdir)
    finally:
        # put back the old replay, if it's been overwritten
        if old_context is not None:
            with open(REPLAY_PATH, 'w') as replay_file:
                json.dump(old_context, replay_file)
    return tempdir / name

def update_project_template():
    cwd = pathlib.Path(os.getcwd())
    try:
        current_repo = git.Repo(safepath(cwd), search_parent_directories=True) # for right now
        context_path = cwd.joinpath('.cookiecutter_replay.json')
        with open(context_path) as context_file:
            context = json.load(context_file)
    except:
        logger.exception('Could not find ballet repo, update failed')

    with tempfile.TemporaryDirectory() as tempdir:
        updated_template = _create_replay(tempdir, context)
        updated_repo = git.Repo(safepath(updated_template))
        # add some randomness in the remote name by using tempdir
        remote_name = updated_template.parts[-1]
        try:
            updated_remote = current_repo.create_remote(remote_name, updated_repo.working_tree_dir)
            updated_remote.fetch()
            current_repo.git.merge(
                remote_name + '/master',
                allow_unrelated_histories=True,
                # strategy_option='theirs',
                squash=True,
            )
        except: 
            logger.exception('Could not merge changes into project, update failed')
        finally:
            current_repo.delete_remote(remote_name)


def main():
    update_project_template()