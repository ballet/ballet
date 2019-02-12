import git
import os
import tempfile
import yaml

from cookiecutter.main import cookiecutter

from ballet.compat import pathlib, safepath
from ballet.util.log import logger
from ballet.project import get_config_paths, find_configs
from ballet.quickstart import generate_project

def _create_replay(tempdir, name):
    generate_project(replay=True, output_dir=tempdir)
    return pathlib.Path(tempdir) / name

def update_project():
    cwd = pathlib.Path(os.getcwd())
    ballet_yml = find_configs(cwd)
    try:
        current_repo = git.Repo(safepath(cwd), search_parent_directories=True) # for right now
    except Exception as e:
        print(e)

    with tempfile.TemporaryDirectory() as tempdir:
        updated_project = _create_replay(tempdir, ballet_yml[0]['problem']['name'])
        updated_repo = git.Repo(safepath(updated_project))
        remote_name = updated_project.parts[-1]
        try:
            # add some randomness in the remote name
            updated_remote = current_repo.create_remote(remote_name, updated_repo.working_tree_dir)
            updated_remote.fetch()
            current_repo.git.merge(
                remote_name + '/master',
                allow_unrelated_histories=True,
                # strategy_option='theirs',
                squash=True,
            )
        except Exception as e: 
            print(e)
            logger.exception('Could not complete update, merge failed')
        finally:
            current_repo.delete_remote(remote_name)


def main():
    update_project()