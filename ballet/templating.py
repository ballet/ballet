import pathlib
import tempfile
from typing import List, Optional, Tuple

import funcy as fy
from cookiecutter.main import cookiecutter as _cookiecutter
from github import Github, GithubException

import ballet.util.git
from ballet.compat import PathLike
from ballet.exc import BalletError, ConfigurationError
from ballet.project import Project, detect_github_username
from ballet.util.fs import pwalk, synctree
from ballet.util.git import (
    DEFAULT_BRANCH, push_branches_to_remote, switch_to_new_branch,)
from ballet.util.log import logger
from ballet.util.typing import Pathy
from ballet.validation.project_structure.checks import (
    FEATURE_MODULE_NAME_REGEX, SUBPACKAGE_NAME_REGEX,)

TEMPLATES_PATH = pathlib.Path(__file__).resolve().parent.joinpath('templates')
FEATURE_TEMPLATE_PATH = TEMPLATES_PATH.joinpath('feature_template')
PROJECT_TEMPLATE_PATH = TEMPLATES_PATH.joinpath('project_template')


def _stringify_path(obj) -> str:
    return str(obj) if isinstance(obj, PathLike) else obj


@fy.wraps(_cookiecutter)
def cookiecutter(*args, **kwargs) -> str:
    """Call cookiecutter.main.cookiecutter after stringifying paths

    Return:
        project directory path
    """
    args = fy.walk(_stringify_path, args)
    kwargs = fy.walk_values(_stringify_path, kwargs)
    return _cookiecutter(*args, **kwargs)


def render_project_template(
    project_template_path: Optional[Pathy] = None,
    create_github_repo: bool = False,
    github_token: Optional[str] = None,
    **cc_kwargs
) -> str:
    """Generate a ballet project according to the project template

    If creating the GitHub repo is requested and the process fails for any
    reason, quickstart will complete successfully and users are instructed
    to read the corresponding section of the Maintainer's Guide to continue
    manually.

    Args:
        project_template_path: path to specific project template
        create_github_repo: whether to act to create the desired repo on
            GitHub after rendering the project. The repo will be owned by
            either the user or an org that the user has relevant permissions
            for, depending on what is entered during the quickstart prompts.
            If True, then a valid github token must also be provided.
        github_token: valid github token with appropriate permissions
        **cc_kwargs: options for the cookiecutter template
    """
    if project_template_path is None:
        project_template_path = PROJECT_TEMPLATE_PATH

    project_path = cookiecutter(project_template_path, **cc_kwargs)

    if create_github_repo:
        if github_token is None:
            raise ValueError('Need to provide github token')
        g = Github(github_token)

        # need to get params from new project config
        project = Project.from_path(project_path)
        owner = project.config.get('github.github_owner')
        name = project.config.get('project.project_slug')

        # create repo on github
        try:
            github_repo = ballet.util.git.create_github_repo(g, owner, name)
            logger.info(f'Created repo on GitHub at {github_repo.html_url}')
        except GithubException:
            logger.exception('Failed to create GitHub repo for this project')
            logger.warning(
                'Failed to create GitHub repo for this project...\n'
                'did you specify the intended repo owner, and do you have'
                ' permissions to create a repo under that owner?\n'
                'Try manually creating the repo: https://ballet.github.io/ballet/maintainer_guide.html#manual-repository-creation'  # noqa E501
            )
            return project_path

        # now push to remote
        # we don't need to set up the remote, as it has already been setup in
        # post_gen_hook.py
        local_repo = project.repo
        remote_name = project.config.get('github.remote')
        branches = [DEFAULT_BRANCH]
        try:
            push_branches_to_remote(local_repo, remote_name, branches)
        except BalletError:
            logger.exception('Failed to push branches to GitHub repo')
            logger.warning(
                'Failed to push branches to GitHub repo...\n'
                'Try manually pushing the branches: https://ballet.github.io/ballet/maintainer_guide.html#manual-repository-creation'  # noqa E501
            )
            return project_path

    return project_path


def render_feature_template(**cc_kwargs) -> str:
    """Create a stub for a new feature

    Args:
        **cc_kwargs: options for the cookiecutter template
    """
    feature_template_path = FEATURE_TEMPLATE_PATH
    return cookiecutter(feature_template_path, **cc_kwargs)


def _fail_if_feature_exists(dst: pathlib.Path) -> None:
    subpackage_name, feature_name = str(dst.parent), str(dst.name)
    if (
        dst.is_file()
        and fy.re_test(SUBPACKAGE_NAME_REGEX, subpackage_name)
        and fy.re_test(FEATURE_MODULE_NAME_REGEX, feature_name)
    ):
        raise FileExistsError(f'The feature already exists here: {dst}')


def start_new_feature(
    contrib_dir: Pathy = None,
    branching: bool = True,
    **cc_kwargs
) -> List[Tuple[pathlib.Path, str]]:
    """Start a new feature within a ballet project

    If run from default branch, by default will attempt to switch to a new
    branch for this feature, given by `<username>/feature-<featurename>`. By
    default, will prompt the user for input using cookiecutter's input
    interface.

    Renders the feature template into a temporary directory, then copies the
    feature files into the proper path within the contrib directory.

    Args:
        contrib_dir: directory under which to place contributed features
        branching: whether to attempt to manage branching
        **cc_kwargs: options for the cookiecutter template

    Raises:
        ballet.exc.BalletError: the new feature has the same name as an
            existing one
    """
    if contrib_dir is not None:
        try:
            project = Project.from_path(contrib_dir, ascend=True)
            default_username = detect_github_username(project)
        except ConfigurationError:
            default_username = 'username'
    else:
        project = Project.from_cwd()
        contrib_dir = project.config.get('contrib.module_path')
        default_username = detect_github_username(project)

    # inject default username into context
    cc_kwargs.setdefault('extra_context', {})
    cc_kwargs['extra_context'].update({'_default_username': default_username})

    with tempfile.TemporaryDirectory() as tempdir:
        # render feature template
        output_dir = tempdir
        cc_kwargs['output_dir'] = output_dir
        rendered_dir = render_feature_template(**cc_kwargs)

        # clean pyc files from rendered dir
        for path in pwalk(rendered_dir, topdown=False):
            if path.suffix == '.pyc':
                path.unlink()
            if path.name == '__pycache__':
                with fy.suppress(OSError):
                    path.rmdir()

        # copy into contrib dir
        src = rendered_dir
        dst = contrib_dir
        result = synctree(src, dst, onexist=_fail_if_feature_exists)

    if branching and project.on_master:
        # what is the target branch?
        target_branch = None
        paths = [path for path, kind in result if kind == 'file']
        for path in paths:
            parts = pathlib.Path(path).parts
            subpackage, module = parts[-2], parts[-1]
            user_match = fy.re_find(SUBPACKAGE_NAME_REGEX, subpackage)
            feature_match = fy.re_find(FEATURE_MODULE_NAME_REGEX, module)
            if feature_match:
                username = user_match['username']
                featurename = feature_match['featurename'].replace('_', '-')
                target_branch = f'{username}/feature-{featurename}'

        if target_branch is not None:
            switch_to_new_branch(project.repo, target_branch)

    _log_start_new_feature_success(result)

    return result


def _log_start_new_feature_success(result: List[Tuple[pathlib.Path, str]]):
    logger.info('Start new feature successful.')
    for (name, kind) in result:
        if kind == 'file' and '__init__' not in str(name):
            relname = pathlib.Path(name).relative_to(pathlib.Path.cwd())
            logger.info(f'Created {relname}')
