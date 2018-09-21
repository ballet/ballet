import os
import pathlib
import tempfile
import types
import unittest
from unittest.mock import patch

import ballet
from ballet.util.gitutil import get_diff_str_from_commits
from ballet.util.modutil import (  # noqa F401
    import_module_at_path, import_module_from_modname,
    import_module_from_relpath, modname_to_relpath, relpath_to_modname)
from ballet.util.travisutil import (
    TravisPullRequestBuildDiffer, get_travis_pr_num, is_travis_pr)

from .util import make_mock_commits, mock_repo


class TestModutil(unittest.TestCase):

    @unittest.expectedFailure
    def test_import_module_from_modname(self):
        raise NotImplementedError

    @unittest.expectedFailure
    def test_import_module_from_relpath(self):
        raise NotImplementedError

    def test_import_module_at_path_module(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir).joinpath('foo', 'bar.py')
            path.parent.mkdir(parents=True)
            init = path.parent.joinpath('__init__.py')
            init.touch()
            x = 1
            with path.open('w') as f:
                f.write('x={x!r}'.format(x=x))
            modname = 'foo.bar'
            modpath = str(path)  # e.g. /tmp/foo/bar.py'
            mod = import_module_at_path(modname, modpath)
            self.assertIsInstance(mod, types.ModuleType)
            self.assertEqual(mod.__name__, modname)
            self.assertEqual(mod.x, x)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir).joinpath('foo')
            path.mkdir(parents=True)
            init = path.joinpath('__init__.py')
            init.touch()
            x = 'hello'
            with init.open('w') as f:
                f.write('x={x!r}'.format(x=x))
            modname = 'foo'
            modpath = str(path)
            mod = import_module_at_path(modname, modpath)
            self.assertIsInstance(mod, types.ModuleType)
            self.assertEqual(mod.__name__, modname)
            self.assertEqual(mod.x, x)

    @unittest.expectedFailure
    def test_import_module_at_path_bad_package_structure(self):
        raise NotImplementedError

    def test_relpath_to_modname(self):
        relpath = 'ballet/util/_util.py'
        expected_modname = 'ballet.util._util'
        actual_modname = relpath_to_modname(relpath)
        self.assertEqual(actual_modname, expected_modname)

        relpath = 'ballet/util/__init__.py'
        expected_modname = 'ballet.util'
        actual_modname = relpath_to_modname(relpath)
        self.assertEqual(actual_modname, expected_modname)

        relpath = 'ballet/foo/bar/baz.zip'
        with self.assertRaises(ValueError):
            relpath_to_modname(relpath)

    def test_modname_to_relpath(self):
        modname = 'ballet.util._util'
        expected_relpath = 'ballet/util/_util.py'
        actual_relpath = modname_to_relpath(modname)
        self.assertEqual(actual_relpath, expected_relpath)

        modname = 'ballet.util'
        # mypackage.__file__ resolves to the __init__.py
        project_root = pathlib.Path(ballet.__file__).parent.parent

        expected_relpath = 'ballet/util/__init__.py'
        actual_relpath = modname_to_relpath(modname, project_root=project_root)
        self.assertEqual(actual_relpath, expected_relpath)

        # without providing project root, behavior is undefined, as we don't
        # know whether the relative path will resolve to a directory

        # within a temporary directory, the relpath *should not* be a dir
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                actual_relpath = modname_to_relpath(modname)
                expected_relpath = 'ballet/util.py'
                self.assertEqual(actual_relpath, expected_relpath)
            finally:
                os.chdir(cwd)

        # from the actual project root, the relpath *should* be a dir
        cwd = os.getcwd()
        try:
            os.chdir(str(project_root))
            actual_relpath = modname_to_relpath(modname)
            expected_relpath = 'ballet/util/__init__.py'
            self.assertEqual(actual_relpath, expected_relpath)
        finally:
            os.chdir(cwd)

        # without add_init
        modname = 'ballet.util'
        add_init = False
        expected_relpath = 'ballet/util'
        actual_relpath = modname_to_relpath(
            modname, project_root=project_root, add_init=add_init)
        self.assertEqual(actual_relpath, expected_relpath)


class TestTravis(unittest.TestCase):
    def setUp(self):
        self.pr_num = 7
        self.commit_range = 'HEAD^..HEAD'
        self.travis_vars = {
            'TRAVIS_PULL_REQUEST': str(self.pr_num),
            'TRAVIS_COMMIT_RANGE': self.commit_range,
        }

    def test_get_travis_pr_num(self):
        # matrix of env name, setting for env, expected result
        matrix = (
            ('TRAVIS_PULL_REQUEST', str(self.pr_num), self.pr_num),
            ('TRAVIS_PULL_REQUEST', 'true', None),
            ('TRAVIS_PULL_REQUEST', 'FALSE', None),
            ('TRAVIS_PULL_REQUEST', 'false', None),
            ('TRAVIS_PULL_REQUEST', 'abcd', None),
            ('UNRELATED_VAR', '', None),
        )
        for env_name, env_value, expected_result in matrix:
            with patch.dict('os.environ', {env_name: env_value}):
                actual_result = get_travis_pr_num()
                self.assertEqual(actual_result, expected_result)

    def test_is_travis_pr(self):
        matrix = (
            ('TRAVIS_PULL_REQUEST', str(self.pr_num), True),
            ('TRAVIS_PULL_REQUEST', 'true', False),
            ('TRAVIS_PULL_REQUEST', 'FALSE', False),
            ('TRAVIS_PULL_REQUEST', 'false', False),
            ('TRAVIS_PULL_REQUEST', 'abcd', False),
            ('UNRELATED_VAR', '', False),
        )
        for env_name, env_value, expected_result in matrix:
            with patch.dict('os.environ', {env_name: env_value}):
                actual_result = is_travis_pr()
                self.assertEqual(actual_result, expected_result)

    def test_travis_pull_request_build_differ(self):
        with mock_repo() as repo:
            pr_num = self.pr_num
            commit_range = 'HEAD^..HEAD'

            travis_env_vars = {
                'TRAVIS_BUILD_DIR': repo.working_tree_dir,
                'TRAVIS_PULL_REQUEST': str(pr_num),
                'TRAVIS_COMMIT_RANGE': commit_range,
            }
            with patch.dict('os.environ', travis_env_vars):
                differ = TravisPullRequestBuildDiffer(pr_num)
                diff_str = differ._get_diff_str()
                self.assertEqual(diff_str, commit_range)

    def test_travis_pull_request_build_differ_on_mock_commits(self):
        n = 10
        i = 0
        pr_num = self.pr_num
        with mock_repo() as repo:
            commits = make_mock_commits(repo, n=n)
            commit_range = get_diff_str_from_commits(commits[i], commits[-1])

            travis_env_vars = {
                'TRAVIS_BUILD_DIR': repo.working_tree_dir,
                'TRAVIS_PULL_REQUEST': str(pr_num),
                'TRAVIS_COMMIT_RANGE': commit_range,
            }
            with patch.dict('os.environ', travis_env_vars):
                differ = TravisPullRequestBuildDiffer(pr_num)
                diff_str = differ._get_diff_str()
                self.assertEqual(diff_str, commit_range)

                diffs = differ.diff()

                # there should be n-1 diff objects, they should show files
                # 1 to n-1
                self.assertEqual(len(diffs), n - 1)
                j = i + 1
                for diff in diffs:
                    self.assertEqual(diff.change_type, 'A')
                    self.assertEqual(diff.b_path, 'file{j}.py'.format(j=j))
                    j += 1
