import json
import pathlib
import tempfile
from textwrap import dedent
from typing import Optional
from unittest.mock import patch

import funcy
import git
import packaging.version
import requests
from cookiecutter.prompt import prompt_for_config
from git import GitCommandError

import ballet
from ballet.exc import BalletError, ConfigurationError
from ballet.project import Project
from ballet.templating import render_project_template
from ballet.util.git import DEFAULT_BRANCH, push_branches_to_remote
from ballet.util.log import logger
from ballet.util.typing import Pathy

PROJECT_CONTEXT_PATH = (
    pathlib.Path(__file__).resolve().parent.joinpath(
        'templates',
        'project_template',
        'cookiecutter.json'))
CONTEXT_FILE_NAME = '.cookiecutter_context.json'
TEMPLATE_BRANCH = 'project-template'
PYPI_PROJECT_JSON_URL = 'https://pypi.org/pypi/{project}/json'


def _get_latest_project_version_string(project: str) -> Optional[str]:
    """Get the latest version of a project according to the PyPI Warehouse API

    For context, the `sampleproject` project returns 10KB of data.

    Returns:
        latest version of `project` or None if something went wrong
    """
    url = PYPI_PROJECT_JSON_URL.format(project=project)
    response = requests.get(url)
    releases = response.json()['releases'].keys()
    return max(releases, key=packaging.version.parse)


@funcy.ignore(Exception)
def _get_latest_ballet_version_string() -> Optional[str]:
    """Get the latest version of ballet according to the PyPI Warehouse API

    This returns on the order of 50KB of data.

    Returns:
        latest version of ballet or None if something went wrong
    """
    return _get_latest_project_version_string(ballet.__name__)


def _check_for_updated_ballet() -> Optional[str]:
    """Return the version of an updated ballet if it is available

    Returns:
        the latest version of ballet available or None if the latest version
        is the same as the currently-installed version
    """
    latest = _get_latest_ballet_version_string()
    current = ballet.__version__
    parse = packaging.version.parse
    if latest and parse(latest) > parse(current):
        return latest
    else:
        return None


def _warn_of_updated_ballet(latest: Optional[str]):
    if latest is not None:
        current = ballet.__version__
        msg = dedent(
            f'''
            A new version of ballet is available: v{latest}
            - you currently have ballet v{current}
            - if you don't update ballet, you won't receive project template updates
            - update ballet and then try again:

                $ python -m pip install --upgrade ballet
            '''  # noqa E501
        ).strip()
        logger.warning(msg)


def _make_template_branch_merge_commit_message() -> str:
    version = ballet.__version__
    return f'Merge project template updates from ballet v{version}'


@funcy.silent
def _safe_delete_remote(repo: git.Repo, name: str):
    repo.delete_remote(name)


def _render_project_template(
    cwd: pathlib.Path,
    tempdir: Pathy,
    project_template_path: Optional[Pathy] = None
) -> str:
    tempdir = pathlib.Path(tempdir)
    context = _get_full_context(cwd)

    # don't dump replay files to home directory.
    with patch('cookiecutter.main.dump'):
        return render_project_template(
            project_template_path=project_template_path,
            no_input=True,
            extra_context=context,
            output_dir=tempdir)


def _get_full_context(cwd: pathlib.Path) -> dict:
    # load the context stored within the project repository
    context_path = cwd.joinpath(CONTEXT_FILE_NAME)
    if context_path.exists():
        with context_path.open('r') as f:
            context = json.load(f)
    else:
        raise FileNotFoundError(
            f'Could not find \'{CONTEXT_FILE_NAME}\', are you in a ballet '
            'project repo?')

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


def _log_recommended_reinstall():
    logger.info(
        'After a successful project template update, try re-installing the\n'
        'project in case the project template requires any different \n'
        'dependencies than what you have installed:\n'
        '\n'
        '    $ invoke install')


def update_project_template(push: bool = False,
                            project_template_path: Optional[Pathy] = None):
    """Update project with updates to upstream project template

    The update is fairly complicated and proceeds as follows:

    1. Load project: user must run command from master branch and ballet
       must be able to detect the project-template branch
    2. Load the saved cookiecutter context from disk
    3. Render the project template into a temporary directory using the
       saved context, *prompting the user if new keys are required*. Note
       that the project template is simply loaded from the data files of the
       installed version of ballet. Note further that by the project
       template's post_gen_hook, a new git repo is initialized [in the
       temporary directory] and files are committed.
    4. Add the temporary directory as a remote and merge it into the
       project-template branch, favoring changes made to the upstream template.
       Any failure to merge results in an unrecoverable error.
    5. Merge the project-template branch into the master branch. The user is
       responsible for merging conflicts and they are given instructions to
       do so and recover.
    6. If applicable, push to master.

    Args:
        push: whether to push updates to remote, defaults to False
        project_template_path: an override for the path to the
            project template
    """
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
        raise BalletError(
            'Can\'t update project template with uncommitted changes. '
            'Please commit your changes and try again.')

    if repo.head.ref.name != DEFAULT_BRANCH:
        raise ConfigurationError(
            f'Must run command from branch {DEFAULT_BRANCH}')

    if TEMPLATE_BRANCH not in repo.branches:
        raise ConfigurationError(
            f'Could not find \'{TEMPLATE_BRANCH}\' branch.')

    # check for upstream updates to ballet
    new_version = _check_for_updated_ballet()
    if new_version:
        _warn_of_updated_ballet(new_version)

    with tempfile.TemporaryDirectory() as _tempdir:
        tempdir = pathlib.Path(_tempdir)

        # cookiecutter returns path to the resulting project dir
        logger.debug(f'Re-rendering project template at {tempdir}')
        updated_template = _render_project_template(
            cwd, tempdir, project_template_path=project_template_path)
        updated_repo = git.Repo(updated_template)

        # tempdir is a randomly-named dir suitable for a random remote name
        # to avoid conflicts
        remote_name = tempdir.name

        remote = repo.create_remote(
            remote_name, updated_repo.working_tree_dir)
        remote.fetch()

        repo.heads[TEMPLATE_BRANCH].checkout()
        try:
            logger.debug('Merging re-rendered template to project-template '
                         'branch')
            repo.git.merge(
                remote_name + '/' + DEFAULT_BRANCH,
                allow_unrelated_histories=True,
                strategy_option='theirs',
                squash=True,
            )
            if not repo.is_dirty():
                logger.info('No updates to template -- done.')
                return
            commit_message = _make_template_branch_merge_commit_message()
            logger.debug(f'Committing updates: {commit_message}')
            repo.git.commit(m=commit_message)
        except GitCommandError:
            logger.critical(
                f'Could not merge changes into {TEMPLATE_BRANCH} branch, '
                f'update failed')
            raise
        finally:
            _safe_delete_remote(repo, remote_name)
            logger.debug('Checking out master branch')
            repo.heads[DEFAULT_BRANCH].checkout()

    try:
        logger.debug('Merging project-template branch into master')
        repo.git.merge(TEMPLATE_BRANCH, no_ff=True)
    except GitCommandError as e:
        if 'merge conflict' in str(e).lower():
            logger.critical(dedent(
                f'''
                Update failed due to a merge conflict.
                Fix conflicts, and then complete merge manually:
                    $ git add .
                    $ git commit --no-edit
                Otherwise, abandon the update:
                    $ git reset --merge {original_head}
                '''
            ).strip())
        raise

    if push:
        repo = project.repo
        remote_name = project.config.get('github.remote')
        branches = [DEFAULT_BRANCH, TEMPLATE_BRANCH]
        push_branches_to_remote(repo, remote_name, branches)

    _log_recommended_reinstall()
