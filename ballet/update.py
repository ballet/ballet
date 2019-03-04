import json
import tempfile
from unittest.mock import patch

import funcy
import git
from cookiecutter.prompt import prompt_for_config
from git import GitCommandError

from ballet import __version__ as version
from ballet.compat import pathlib, safepath
from ballet.exc import ConfigurationError, Error
from ballet.project import Project
from ballet.quickstart import generate_project
from ballet.util.log import logger

PROJECT_CONTEXT_PATH = (
    pathlib.Path(__file__).resolve().parent.joinpath(
        'project_template',
        'cookiecutter.json'))
CONTEXT_FILE_NAME = '.cookiecutter_context.json'
TEMPLATE_BRANCH = 'project-template'


def _make_template_branch_merge_commit_message():
    return 'Merge project template updates from ballet v{}'.format(version)


def _safe_delete_remote(repo, name):
    with funcy.suppress(Exception):
        repo.delete_remote(name)


def _render_project_template(cwd, tempdir):
    tempdir = pathlib.Path(tempdir)
    context = _get_full_context(cwd)

    # don't dump replay files to home directory.
    with patch('cookiecutter.main.dump'):
        return generate_project(no_input=True,
                                extra_context=context,
                                output_dir=safepath(tempdir))


def _get_full_context(cwd):
    # load the context stored within the project repository
    context_path = cwd.joinpath(CONTEXT_FILE_NAME)
    if context_path.exists():
        with context_path.open('r') as f:
            context = json.load(f)
    else:
        raise FileNotFoundError(
            'Could not find \'{}\', are you in a ballet project repo?'
            .format(CONTEXT_FILE_NAME))

    # find out if there are any new keys to prompt for
    with PROJECT_CONTEXT_PATH.open('r') as f:
        new_context = json.load(f)
    new_keys = set(new_context) - set(context['cookiecutter'])
    if new_keys:
        new_context_config = {'cookiecutter': funcy.project(new_context,
                                                            new_keys)}
        new_context = prompt_for_config(new_context_config)
        context['cookiecutter'].update(new_context)

    return context['cookiecutter']


def update_project_template():
    cwd = pathlib.Path.cwd().resolve()

    # get ballet project info -- must be at project root directory with a
    # ballet.yml file.
    try:
        project = Project.from_path(cwd)
    except ConfigurationError:
        raise ConfigurationError('Must run command from project root.')

    repo = project.repo
    original_head = repo.head.commit.hexsha[:7]

    if repo.is_dirty():
        raise Error(
            'Can\'t update project template with uncommitted changes. '
            'Please commit your changes and try again.')

    if TEMPLATE_BRANCH not in repo.branches:
        raise ConfigurationError(
            'Could not find \'{}\' branch.'.format(TEMPLATE_BRANCH))

    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = pathlib.Path(tempdir)
        updated_template = _render_project_template(cwd, tempdir)
        updated_repo = git.Repo(safepath(updated_template))

        # tempdir is a randomly-named dir
        remote_name = tempdir.parts[-1]

        remote = repo.create_remote(
            remote_name, updated_repo.working_tree_dir)
        remote.fetch()

        repo.heads[TEMPLATE_BRANCH].checkout()
        try:
            repo.git.merge(
                remote_name + '/master',
                allow_unrelated_histories=True,
                strategy_option='theirs',
                squash=True,
            )
            if not repo.is_dirty():
                logger.info('No updates to template -- done.')
                return
            commit_message = _make_template_branch_merge_commit_message()
            repo.git.commit(m=commit_message)
        except GitCommandError:
            logger.exception(
                'Could not merge changes into {template_branch} branch, '
                'update failed'
                .format(template_branch=TEMPLATE_BRANCH))
            raise
        finally:
            _safe_delete_remote(repo, remote_name)
            repo.heads.master.checkout()

    try:
        repo.git.merge(TEMPLATE_BRANCH, no_ff=True)
    except GitCommandError as e:
        if 'merge conflict' in str(e).lower():
            logger.info('\n'.join([
                'Update failed due to a merge conflict.',
                'Fix conflicts, and then complete merge manually:',
                '    $ git add .',
                '    $ git commit --no-edit',
                'Otherwise, abandon the update:',
                '    $ git reset --merge {original_head}'
            ]).format(original_head=original_head))
        raise

    logger.info('Update successful.')


def main():
    import ballet.util.log
    ballet.util.log.enable(level='INFO', echo=False)
    update_project_template()
