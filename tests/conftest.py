import shutil
import tempfile
from collections import namedtuple
from unittest.mock import patch

import git
import pytest
from cookiecutter.utils import work_in

import ballet
from ballet.compat import pathlib, safepath
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
    with work_in(safepath(tempdir)):

        project_slug = 'foo'
        extra_context = {
            'project_slug': project_slug,
        }

        # ballet quickstart
        render_project_template(no_input=True,
                                extra_context=extra_context,
                                output_dir=safepath(tempdir))

        # tree .
        tree(tempdir)

        repo = git.Repo(safepath(tempdir.joinpath(project_slug)))

        yield (
            namedtuple('Quickstart', 'tempdir project_slug repo')
            ._make((tempdir, project_slug, repo))
        )


@pytest.fixture
def project_template_copy(tempdir):
    old_path = ballet.templating._get_project_template_path()
    new_path = tempdir.joinpath('templates', 'project_template')
    shutil.copytree(old_path, safepath(new_path))

    with patch('ballet.templating._get_project_template_path') as m:
        m.return_value = str(new_path)
        tree(new_path)
        yield new_path
