import pathlib
import tempfile
from typing import List, Optional, Tuple

import funcy as fy
from cookiecutter.main import cookiecutter as _cookiecutter

from ballet.compat import PathLike
from ballet.exc import ConfigurationError
from ballet.project import Project, detect_github_username
from ballet.util.fs import pwalk, synctree
from ballet.util.log import logger
from ballet.util.typing import Pathy
from ballet.validation.project_structure.checks import (
    FEATURE_MODULE_NAME_REGEX, SUBPACKAGE_NAME_REGEX)

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
    project_template_path: Optional[Pathy] = None, **cc_kwargs
) -> str:
    """Generate a ballet project according to the project template

    Args:
        project_template_path: path to specific project template
        **cc_kwargs: options for the cookiecutter template
    """
    if project_template_path is None:
        project_template_path = PROJECT_TEMPLATE_PATH
    return cookiecutter(project_template_path, **cc_kwargs)


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
        raise FileExistsError(
            'The feature already exists here: {dst}'
            .format(dst=dst))


def start_new_feature(
    contrib_dir: Pathy = None, **cc_kwargs
) -> List[Tuple[pathlib.Path, str]]:
    """Start a new feature within a ballet project

    By default, will prompt the user for input using cookiecutter's input
    interface.

    Renders the feature template into a temporary directory, then copies the
    feature files into the proper path within the contrib directory.

    Args:
        contrib_dir: directory under which to place contributed features
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

    _log_start_new_feature_success(result)

    return result


def _log_start_new_feature_success(result: List[Tuple[pathlib.Path, str]]):
    logger.info('Start new feature successful.')
    for (name, kind) in result:
        if kind == 'file' and '__init__' not in str(name):
            relname = pathlib.Path(name).relative_to(pathlib.Path.cwd())
            logger.info('Created {}'.format(relname))
