import pathlib
import shutil
import tempfile
from typing import NamedTuple
from unittest.mock import patch

import git
import pytest
import responses as _responses

import ballet
from ballet.project import Project
from ballet.templating import render_project_template
from ballet.util import work_in

from .util import set_ci_git_config_variables, tree


@pytest.fixture
def tempdir():
    """Tempdir fixture using tempfile.TemporaryDirectory"""
    with tempfile.TemporaryDirectory() as d:
        yield pathlib.Path(d)


@pytest.fixture
def testdatadir():
    return pathlib.Path(__file__).resolve().parent.joinpath('testdata')


def _mock_repo(tempdir):
    repo = git.Repo.init(str(tempdir))
    set_ci_git_config_variables(repo)
    return repo


@pytest.fixture
def mock_repo(tempdir):
    """Create a new repo"""
    yield _mock_repo(tempdir)


class QuickstartResult(NamedTuple):
    project: Project
    tempdir: pathlib.Path
    project_slug: str
    package_slug: str
    repo: git.Repo


@pytest.fixture
def quickstart(tempdir):
    """
    $ cd tempdir
    $ ballet quickstart
    $ tree .
    """
    # cd tempdir
    with work_in(tempdir):

        project_slug = 'foo-bar'
        package_slug = 'foo_bar'
        extra_context = {
            'project_name': project_slug.capitalize(),
            'project_slug': project_slug,
            'package_slug': package_slug,
        }

        # ballet quickstart
        render_project_template(no_input=True,
                                extra_context=extra_context,
                                output_dir=tempdir)

        # tree .
        tree(tempdir)

        project = Project.from_path(tempdir.joinpath(project_slug))
        repo = project.repo

        yield QuickstartResult(
            project, tempdir, project_slug, package_slug, repo
        )


@pytest.fixture
def project_template_copy(tempdir):
    old_path = ballet.templating.PROJECT_TEMPLATE_PATH
    new_path = tempdir.joinpath('templates', 'project_template')
    shutil.copytree(old_path, new_path)

    with patch('ballet.templating.PROJECT_TEMPLATE_PATH', new_path):
        tree(new_path)
        yield new_path


@pytest.fixture
def responses():
    with _responses.RequestsMock() as rsps:
        yield rsps
