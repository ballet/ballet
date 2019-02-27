import os
import shutil
import subprocess
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


def _tree(dir):
    with funcy.suppress((FileNotFoundError, subprocess.SubprocessError)):
        cmd = ['tree', '-A', '-n', '--charset', 'ASCII', str(dir)]
        logger.debug('Popen({cmd!r})'.format(cmd=cmd))
        tree_output = subprocess.check_output(cmd).decode()
        logger.debug(tree_output)


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
        _tree(tmpdir)

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
        _tree(new_path)
        yield new_path


def _run_ballet_update_template(d, project_slug):
    with chdir(safepath(d.joinpath(project_slug))):
        ballet.update.update_project_template(create_merge_commit=True)


@pytest.mark.usefixtures('clean_system')
def test_update_after_change_in_template(quickstart, project_template_copy):
    """
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

    # add foo: bar to project template
    with template_dir.joinpath(
            '{{cookiecutter.project_slug}}', 'ballet.yml').open('a') as f:
        f.write('\nfoo: bar\n')

    # run ballet-update-template
    _run_ballet_update_template(tmpdir, project_slug)

    # run git log
    git_log_output = repo.git.log(n=2)
    logger.debug(git_log_output)

    # assert on branch master
    expected_branch = 'master'
    actual_branch = repo.head.ref.name

    assert actual_branch == expected_branch

    # assert commit message is automatically generated
    expected_commit_message = \
        ballet.update._make_master_branch_merge_commit_message().strip()
    actual_commit_message = repo.head.commit.message.strip()

    assert actual_commit_message == expected_commit_message

    # assert foo: bar is in ballet.yml
    with tmpdir.joinpath(project_slug, 'ballet.yml').open('r') as f:
        lines = f.readlines()
        last_line = lines[-1]
        assert 'foo: bar' in last_line


@pytest.mark.usefixtures('clean_system')
def test_update_after_change_in_project(quickstart):
    """
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

    # add foo: bar
    with tmpdir.joinpath(project_slug, 'ballet.yml').open('a') as f:
        f.write('\nfoo: bar\n')

    # commit
    repo.git.add('ballet.yml')
    repo.git.commit(m='Add "foo: bar" to ballet.yml')

    # run ballet-update-template
    _run_ballet_update_template(tmpdir, project_slug)

    # run git log
    git_log_output = repo.git.log(n=2)
    logger.debug(git_log_output)

    # assert on branch master
    expected_branch = 'master'
    actual_branch = repo.head.ref.name

    assert actual_branch == expected_branch

    # assert commit message is automatically generated
    expected_commit_message = \
        ballet.update._make_master_branch_merge_commit_message().strip()
    actual_commit_message = repo.head.commit.message.strip()

    assert actual_commit_message == expected_commit_message

    # assert foo: bar is in ballet.yml
    with tmpdir.joinpath(project_slug, 'ballet.yml').open('r') as f:
        lines = f.readlines()
        last_line = lines[-1]
        assert 'foo: bar' in last_line


@pytest.mark.xfail
@pytest.mark.usefixtures('clean_system')
def test_update_after_conflicting_changes(quickstart, project_template_copy):
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
@pytest.mark.usefixtures('clean_system')
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

    actual_branch = repo.head.ref.name
    expected_branch = 'master'
    assert actual_branch == expected_branch

    try:
        actual_commit = repo.head.commit
        assert actual_commit == expected_commit
    except AssertionError:
        git_log_output = repo.git.log()
        logger.debug(git_log_output)
        raise
