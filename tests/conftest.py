import pathlib
import shutil
import tempfile
from collections import namedtuple
from unittest.mock import patch

import pytest
from cookiecutter.utils import work_in

import ballet
from ballet.project import Project
from ballet.templating import render_project_template
from tests.util import tree


@pytest.fixture
def tempdir():
    """Tempdir fixture using tempfile.TemporaryDirectory"""
    with tempfile.TemporaryDirectory() as d:
        yield pathlib.Path(d)


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

        yield (
            namedtuple('Quickstart',
                       'project tempdir project_slug package_slug repo')
            ._make((project, tempdir, project_slug, package_slug, repo))
        )


@pytest.fixture
def project_template_copy(tempdir):
    old_path = ballet.templating.PROJECT_TEMPLATE_PATH
    new_path = tempdir.joinpath('templates', 'project_template')
    shutil.copytree(old_path, new_path)

    with patch('ballet.templating.PROJECT_TEMPLATE_PATH', new_path):
        tree(new_path)
        yield new_path
