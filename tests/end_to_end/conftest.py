# Copyright (c) 2013-2017, Audrey Roy
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or
# without modification, are permitted provided that the following
# conditions are met:
#
# * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided
# with the distribution.
#
# * Neither the name of border nor the names of its contributors
# may be used to endorse or promote products derived from this
# software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Source: https://github.com/audreyr/cookiecutter/blob/1.6.0/tests/conftest.py
"""

import os
import shutil

import pytest
from cookiecutter import utils


def backup_dir(original_dir, backup_dir):
    # If the default original_dir is pre-existing, move it to a temp location
    if not os.path.isdir(original_dir):
        return False

    # Remove existing backups before backing up. If they exist, they're stale.
    if os.path.isdir(backup_dir):
        utils.rmtree(backup_dir)

    shutil.copytree(original_dir, backup_dir)
    return True


def restore_backup_dir(original_dir, backup_dir, original_dir_found):
    # Carefully delete the created original_dir only in certain
    # conditions.
    original_dir_is_dir = os.path.isdir(original_dir)
    if original_dir_found:
        # Delete the created original_dir as long as a backup
        # exists
        if original_dir_is_dir and os.path.isdir(backup_dir):
            utils.rmtree(original_dir)
    else:
        # Delete the created original_dir.
        # There's no backup because it never existed
        if original_dir_is_dir:
            utils.rmtree(original_dir)

    # Restore the user's default original_dir contents
    if os.path.isdir(backup_dir):
        shutil.copytree(backup_dir, original_dir)
    if os.path.isdir(original_dir):
        utils.rmtree(backup_dir)


@pytest.fixture(scope='function')
def clean_system(request):
    """
    Fixture that simulates a clean system with no config/cloned cookiecutters.
    It runs code which can be regarded as setup code as known from a unittest
    TestCase. Additionally it defines a local function referring to values
    which have been stored to local variables in the setup such as the location
    of the cookiecutters on disk. This function is registered as a teardown
    hook with `request.addfinalizer` at the very end of the fixture. Pytest
    runs the named hook as soon as the fixture is out of scope, when the test
    finished to put it another way.
    During setup:
    * Back up the `~/.cookiecutterrc` config file to `~/.cookiecutterrc.backup`
    * Back up the `~/.cookiecutters/` dir to `~/.cookiecutters.backup/`
    * Back up the `~/.cookiecutter_replay/` dir to
      `~/.cookiecutter_replay.backup/`
    * Starts off a test case with no pre-existing `~/.cookiecutterrc` or
      `~/.cookiecutters/` or `~/.cookiecutter_replay/`
    During teardown:
    * Delete `~/.cookiecutters/` only if a backup is present at
      `~/.cookiecutters.backup/`
    * Delete `~/.cookiecutter_replay/` only if a backup is present at
      `~/.cookiecutter_replay.backup/`
    * Restore the `~/.cookiecutterrc` config file from
      `~/.cookiecutterrc.backup`
    * Restore the `~/.cookiecutters/` dir from `~/.cookiecutters.backup/`
    * Restore the `~/.cookiecutter_replay/` dir from
      `~/.cookiecutter_replay.backup/`
    """

    # If ~/.cookiecutterrc is pre-existing, move it to a temp location
    user_config_path = os.path.expanduser('~/.cookiecutterrc')
    user_config_path_backup = os.path.expanduser(
        '~/.cookiecutterrc.backup'
    )
    if os.path.exists(user_config_path):
        user_config_found = True
        shutil.copy(user_config_path, user_config_path_backup)
        os.remove(user_config_path)
    else:
        user_config_found = False

    # If the default cookiecutters_dir is pre-existing, move it to a
    # temp location
    cookiecutters_dir = os.path.expanduser('~/.cookiecutters')
    cookiecutters_dir_backup = os.path.expanduser('~/.cookiecutters.backup')
    cookiecutters_dir_found = backup_dir(
        cookiecutters_dir, cookiecutters_dir_backup
    )

    # If the default cookiecutter_replay_dir is pre-existing, move it to a
    # temp location
    cookiecutter_replay_dir = os.path.expanduser('~/.cookiecutter_replay')
    cookiecutter_replay_dir_backup = os.path.expanduser(
        '~/.cookiecutter_replay.backup'
    )
    cookiecutter_replay_dir_found = backup_dir(
        cookiecutter_replay_dir, cookiecutter_replay_dir_backup
    )

    def restore_backup():
        # If it existed, restore ~/.cookiecutterrc
        # We never write to ~/.cookiecutterrc, so this logic is simpler.
        if user_config_found and os.path.exists(user_config_path_backup):
            shutil.copy(user_config_path_backup, user_config_path)
            os.remove(user_config_path_backup)

        # Carefully delete the created ~/.cookiecutters dir only in certain
        # conditions.
        restore_backup_dir(
            cookiecutters_dir,
            cookiecutters_dir_backup,
            cookiecutters_dir_found
        )

        # Carefully delete the created ~/.cookiecutter_replay dir only in
        # certain conditions.
        restore_backup_dir(
            cookiecutter_replay_dir,
            cookiecutter_replay_dir_backup,
            cookiecutter_replay_dir_found
        )

    request.addfinalizer(restore_backup)
