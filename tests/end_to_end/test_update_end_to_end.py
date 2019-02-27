import os
import shutil
from collections import namedtuple
from unittest.mock import patch

import funcy
import git
import pytest

import ballet.quickstart
import ballet.update
from ballet.compat import safepath
from ballet.quickstart import generate_project
from ballet.util.log import logger
from tests.util import tree

pytestmark = pytest.mark.usefixtures('clean_system')


@funcy.contextmanager
def chdir(d):
    olddir = os.getcwd()
    os.chdir(safepath(d))
    yield
    os.chdir(safepath(olddir))


@pytest.fixture
def quickstart(tmp_path):
    """
    $ cd tmpdir
    $ ballet-quickstart
    $ tree .
    """
    tmpdir = tmp_path

    # cd tmpdir
    with chdir(tmpdir):

        project_slug = 'foo'
        extra_context = {
            'project_slug': project_slug,
        }

        # ballet-quickstart
        generate_project(no_input=True,
                         extra_context=extra_context,
                         output_dir=safepath(tmpdir))

        # tree .
        tree(tmpdir)

        repo = git.Repo(safepath(tmpdir.joinpath(project_slug)))

        yield (
            namedtuple('Quickstart', 'tmpdir project_slug repo')
            ._make((tmpdir, project_slug, repo))
        )


@pytest.fixture
def project_template_copy(tmp_path):
    old_path = ballet.quickstart._get_project_template_path()
    new_path = tmp_path.joinpath('project_template')
    shutil.copytree(old_path, safepath(new_path))

    with patch('ballet.quickstart._get_project_template_path') as m:
        m.return_value = str(new_path)
        tree(new_path)
        yield new_path


# ----- Utility methods

def _run_ballet_update_template(d, project_slug):
    with chdir(safepath(d.joinpath(project_slug))):
        ballet.update.update_project_template(create_merge_commit=True)


def check_branch(repo, expected_branch='master'):
    actual_branch = repo.head.ref.name
    assert actual_branch == expected_branch


def check_commit_message(repo):
    expected_commit_message = \
        ballet.update._make_master_branch_merge_commit_message().strip()
    actual_commit_message = repo.head.commit.message.strip()
    assert actual_commit_message == expected_commit_message


def check_modified_file(path, content):
    with path.open('r') as f:
        lines = f.readlines()
        last_line = lines[-1]
        assert content in last_line


def test_update_after_change_in_template(quickstart, project_template_copy):
    """Test update after changing a file in the template

    We expect the project-template branch to be updated and then successfully
    merged into master branch without merge conflicts, and for the changes from
    the template to be reflected in the file on master.

    Setup:

        # quickstart
        # copy project template for modification

    Replicate the following script:

        $ echo '\nfoo: bar' >> /path/to/project_template/ballet.yml
        $ ballet-update-template

    Then test the following:
        - on branch master
        - commit msg on master matches expected msg
        - foo: bar is the last line of ballet.yml
    """
    tmpdir = quickstart.tmpdir
    project_slug = quickstart.project_slug
    repo = quickstart.repo

    template_dir = project_template_copy

    modified_file_path = tmpdir.joinpath(project_slug, 'ballet.yml')
    new_content = 'foo: bar'

    # add foo: bar to project template
    with template_dir.joinpath(
            '{{cookiecutter.project_slug}}', 'ballet.yml').open('a') as f:
        f.write('\n')
        f.write(new_content)
        f.write('\n')

    # run ballet-update-template
    _run_ballet_update_template(tmpdir, project_slug)

    # assert on branch master
    check_branch(repo)

    # assert commit message is automatically generated
    check_commit_message(repo)

    # assert foo: bar is in ballet.yml
    check_modified_file(modified_file_path, new_content)


def test_update_after_change_in_project(quickstart):
    """Test update after changing a file in the project itself

    We expect the updated template to be seamlessly merged, with no merge
    conflicts, and the change to continue to be reflected in the file.

    Setup:

        # quickstart

    Replicate the following script

        $ echo '\nfoo: bar' >> ./ballet.yml
        $ git add ./ballet.yml
        $ git commit -m 'Add "foo: bar" to ballet.yml'
        $ ballet-update-template

    Then test the following:
        - on branch master
        - commit msg on master matches expected automatically-generated msg
        - foo: bar is the last line of ballet.yml
    """
    tmpdir = quickstart.tmpdir
    project_slug = quickstart.project_slug
    repo = quickstart.repo

    modified_file_path = tmpdir.joinpath(project_slug, 'ballet.yml')
    new_content = 'foo: bar'
    # add foo: bar
    with modified_file_path.open('a') as f:
        f.write('\n')
        f.write(new_content)
        f.write('\n')

    # commit
    repo.git.add('ballet.yml')
    repo.git.commit(m='Add "{}" to ballet.yml'.format(new_content))

    # run ballet-update-template
    _run_ballet_update_template(tmpdir, project_slug)

    # assert on branch master
    check_branch(repo)

    # assert commit message is automatically generated
    check_commit_message(repo)

    # assert foo: bar is in ballet.yml
    check_modified_file(modified_file_path, new_content)


@pytest.mark.xfail
def test_update_after_conflicting_changes(quickstart, project_template_copy):
    """Test what happens when there are conflicting changes

    We expect the merge between project-template and master branches to fail in
    some way, whether that is a Python error or just a message to the user, it
    is not decided yet.

    """
    tmpdir = quickstart.tmpdir
    project_slug = quickstart.project_slug
    repo = quickstart.repo

    # add foo: bar
    with tmpdir.joinpath(project_slug, 'ballet.yml').open('a') as f:
        f.write('\nfoo: bar\n')

    # commit
    repo.git.add('ballet.yml')
    repo.git.commit(m='Add "foo: bar" to ballet.yml')

    # add foo: qux to project template
    template_dir = project_template_copy
    with template_dir.joinpath(
            '{{cookiecutter.project_slug}}', 'ballet.yml').open('a') as f:
        f.write('\nfoo: qux\n')

    # run ballet-update-template -- this should raise an error, perhaps,
    # but it doesn't?
    _run_ballet_update_template(tmpdir, project_slug)

    assert False


@pytest.mark.xfail
def test_update_after_no_changes(quickstart):
    """Test that if there are no changes to project template, nothing happens.

    Currently, this test fails because an empty merge commit is created and
    an empty project template update commit is created.

    Setup:

        # quickstart

    Replicate the following script

        $ ballet-update-template

    Then test the following:
        - on branch master
        - no new commit/changes
    """
    tmpdir = quickstart.tmpdir
    project_slug = quickstart.project_slug
    repo = quickstart.repo

    expected_commit = repo.head.commit

    _run_ballet_update_template(tmpdir, project_slug)

    # assert on branch master
    check_branch(repo)

    # assert no new commit/changes
    try:
        actual_commit = repo.head.commit
        assert actual_commit == expected_commit
    except AssertionError:
        git_log_output = repo.git.log()
        logger.debug(git_log_output)
        raise
