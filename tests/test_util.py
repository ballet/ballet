import os
import tempfile
import types
import unittest
from unittest.mock import ANY, mock_open, patch

import numpy as np
import pandas as pd
from funcy import identity

import ballet
import ballet.util.fs
import ballet.util.io
from ballet.compat import pathlib, safepath
from ballet.util.ci import (
    TravisPullRequestBuildDiffer, get_travis_pr_num, is_travis_pr)
from ballet.util.git import (
    get_diff_str_from_commits, get_pull_request_outcomes, get_pull_requests)
from ballet.util.mod import (  # noqa F401
    import_module_at_path, import_module_from_modname,
    import_module_from_relpath, modname_to_relpath, relpath_to_modname)

from .util import make_mock_commits, mock_repo


class UtilTest(unittest.TestCase):

    @unittest.expectedFailure
    def test_asarray2d(self):
        raise NotImplementedError

    @unittest.expectedFailure
    def test_get_arr_desc(self):
        raise NotImplementedError

    @unittest.expectedFailure
    def test_indent(self):
        raise NotImplementedError

    @unittest.expectedFailure
    def test_make_plural_suffix(self):
        raise NotImplementedError

    @unittest.expectedFailure
    def test_whether_failures(self):
        raise NotImplementedError

    def test_has_nans(self):
        objs_with_nans = [
            pd.DataFrame(data={'x': [1, np.nan], 'y': [np.nan, 2]}),
            pd.DataFrame(data={'x': [1, np.nan]}),
            pd.Series(data=[1, np.nan]),
            np.array([[1, np.nan], [np.nan, 2]]),
            np.array([[1, np.nan]]),
            np.array([1, np.nan]),
            np.array([1, np.nan]).T,
            np.array(np.nan),
        ]

        objs_without_nans = [
            pd.DataFrame(data={'x': [1, 0], 'y': [0, 2]}),
            pd.DataFrame(data={'x': [1, 0]}),
            pd.Series(data=[1, 0]),
            np.array([[1, 0], [0, 2]]),
            np.array([[1, 0]]),
            np.array([1, 0]),
            np.array([1, 0]).T,
            np.array(0),
        ]

        for obj in objs_with_nans:
            self.assertTrue(ballet.util.has_nans(obj))

        for obj in objs_without_nans:
            self.assertFalse(ballet.util.has_nans(obj))

    @unittest.expectedFailure
    def test_deepcopy_mixin(self):
        raise NotImplementedError


class ModTest(unittest.TestCase):

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

        # TODO patch this
        # # without providing project root, behavior is undefined, as we don't
        # # know whether the relative path will resolve to a directory

        # # within a temporary directory, the relpath *should not* be a dir
        # with tempfile.TemporaryDirectory() as tmpdir:
        #     cwd = os.getcwd()
        #     try:
        #         os.chdir(tmpdir)
        #         actual_relpath = modname_to_relpath(modname)
        #         expected_relpath = 'ballet/util.py'
        #         self.assertEqual(actual_relpath, expected_relpath)
        #     finally:
        #         os.chdir(cwd)

        # # from the actual project root, the relpath *should* be a dir
        # cwd = os.getcwd()
        # try:
        #     os.chdir(str(project_root))
        #     actual_relpath = modname_to_relpath(modname)
        #     expected_relpath = 'ballet/util/__init__.py'
        #     self.assertEqual(actual_relpath, expected_relpath)
        # finally:
        #     os.chdir(cwd)

        # without add_init
        modname = 'ballet.util'
        add_init = False
        expected_relpath = 'ballet/util'
        actual_relpath = modname_to_relpath(
            modname, project_root=project_root, add_init=add_init)
        self.assertEqual(actual_relpath, expected_relpath)


class CiTest(unittest.TestCase):
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
            ('UNRELATED', '', None),
        )
        for env_name, env_value, expected in matrix:
            with patch.dict('os.environ', {env_name: env_value}, clear=True):
                actual = get_travis_pr_num()
                self.assertEqual(actual, expected)

    def test_is_travis_pr(self):
        matrix = (
            ('TRAVIS_PULL_REQUEST', str(self.pr_num), True),
            ('TRAVIS_PULL_REQUEST', 'true', False),
            ('TRAVIS_PULL_REQUEST', 'FALSE', False),
            ('TRAVIS_PULL_REQUEST', 'false', False),
            ('TRAVIS_PULL_REQUEST', 'abcd', False),
            ('UNRELATED', '', False),
        )
        for env_name, env_value, expected in matrix:
            with patch.dict('os.environ', {env_name: env_value}, clear=True):
                actual = is_travis_pr()
                self.assertEqual(actual, expected)

    def test_travis_pull_request_build_differ(self):
        with mock_repo() as repo:
            pr_num = self.pr_num
            commit_range = 'HEAD^..HEAD'

            travis_env_vars = {
                'TRAVIS_BUILD_DIR': repo.working_tree_dir,
                'TRAVIS_PULL_REQUEST': str(pr_num),
                'TRAVIS_COMMIT_RANGE': commit_range,
            }
            with patch.dict('os.environ', travis_env_vars, clear=True):
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
            with patch.dict('os.environ', travis_env_vars, clear=True):
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


class FsTest(unittest.TestCase):

    def setUp(self):
        self.fs_path_conversions = [identity, pathlib.Path]

    def test_spliceext(self):
        filepath0 = '/foo/bar/baz.txt'

        for func in self.fs_path_conversions:
            filepath = func(filepath0)
            s = '_new'
            expected = '/foo/bar/baz_new.txt'

            actual = ballet.util.fs.spliceext(filepath, s)
            self.assertEqual(actual, expected)

    def test_replaceext(self):
        filepath0 = '/foo/bar/baz.txt'
        expected = '/foo/bar/baz.py'

        for func in self.fs_path_conversions:
            filepath = func(filepath0)

            new_ext = 'py'
            actual = ballet.util.fs.replaceext(filepath, new_ext)
            self.assertEqual(actual, expected)

            new_ext = '.py'
            actual = ballet.util.fs.replaceext(filepath, new_ext)
            self.assertEqual(actual, expected)

    def test_splitext2(self):
        filepath0 = '/foo/bar/baz.txt'
        expected = ('/foo/bar', 'baz', '.txt')

        for func in self.fs_path_conversions:
            filepath = func(filepath0)

            actual = ballet.util.fs.splitext2(filepath)
            self.assertEqual(actual, expected)

        filepath0 = 'baz.txt'
        expected = ('', 'baz', '.txt')

        for func in self.fs_path_conversions:
            filepath = func(filepath0)

            actual = ballet.util.fs.splitext2(filepath)
            self.assertEqual(actual, expected)

    @unittest.expectedFailure
    def test_isemptyfile(self):
        raise NotImplementedError


class GitTest(unittest.TestCase):

    @unittest.expectedFailure
    def test_get_diffs_by_revision(self):
        raise NotImplementedError

    @unittest.expectedFailure
    def test_get_diff_str_from_commits(self):
        raise NotImplementedError

    @unittest.expectedFailure
    def test_get_diffs_by_diff_str(self):
        raise NotImplementedError

    @unittest.expectedFailure
    def test_get_pr_num(self):
        raise NotImplementedError

    @unittest.expectedFailure
    def test_switch_to_new_branch(self):
        raise NotImplementedError

    @unittest.expectedFailure
    def test_set_config_variables(self):
        raise NotImplementedError

    @patch('requests.get')
    def test_get_pull_requests(self, mock_requests_get):
        owner = 'foo'
        repo = 'bar'
        state = 'closed'
        get_pull_requests(owner, repo, state=state)

        (url, ), kwargs = mock_requests_get.call_args
        self.assertIn(owner, url)
        self.assertIn(repo, url)
        self.assertIn('headers', kwargs)
        self.assertIn('params', kwargs)
        self.assertEqual(kwargs['params']['state'], state)

    @patch('ballet.util.git.get_pull_requests')
    def test_get_pull_request_outcomes(self, mock_get_pull_requests):
        mock_get_pull_requests.return_value = [
            {
                'id': 1,
                "created_at": "2011-01-26T19:01:12Z",
                "updated_at": "2011-01-26T19:01:12Z",
                "closed_at": "2011-01-26T19:01:12Z",
                "merged_at": "2011-01-26T19:01:12Z",
            },
            {
                "created_at": "2011-01-26T19:03:19Z",
                "updated_at": "2011-01-26T19:03:19Z",
                "closed_at": "2011-01-26T19:04:01Z",
                "merged_at": None,
            }
        ]
        owner = 'foo'
        repo = 'bar'

        expected = ['accepted', 'rejected']
        actual = get_pull_request_outcomes(owner, repo)
        self.assertEqual(actual, expected)
        mock_get_pull_requests.assert_called_once_with(
            owner, repo, state='closed')


class IoTest(unittest.TestCase):

    def setUp(self):
        self.array = np.arange(10).reshape(2, 5)
        self.frame = pd.util.testing.makeDataFrame()

    def test_check_ext_valid(self):
        ext = '.py'
        expected = ext
        ballet.util.io._check_ext(ext, expected)

    def test_check_ext_invalid_throws(self):
        ext = '.py'
        expected = '.txt'
        with self.assertRaises(ValueError):
            ballet.util.io._check_ext(ext, expected)

    @patch('ballet.util.io._write_tabular_pickle')
    @patch('ballet.util.io._write_tabular_h5')
    def test_write_tabular(self, mock_write_tabular_h5,
                           mock_write_tabular_pickle):
        obj = object()
        filepath = '/foo/bar/baz.h5'
        ballet.util.io.write_tabular(obj, filepath)
        mock_write_tabular_h5.assert_called_once_with(obj, filepath)

        obj = object()
        filepath = '/foo/bar/baz.pkl'
        ballet.util.io.write_tabular(obj, filepath)
        mock_write_tabular_pickle.assert_called_once_with(obj, filepath)

        obj = object()
        filepath = '/foo/bar/baz.xyz'
        with self.assertRaises(NotImplementedError):
            ballet.util.io.write_tabular(obj, filepath)

    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.dump')
    def test_write_tabular_pickle_ndarray(self, mock_dump, mock_open):
        obj = self.array
        filepath = '/foo/bar/baz.pkl'
        ballet.util.io._write_tabular_pickle(obj, filepath)
        mock_dump.assert_called_with(obj, ANY)

    @patch('builtins.open', new_callable=mock_open)
    def test_write_tabular_pickle_ndframe(self, mock_open):
        obj = self.frame
        filepath = '/foo/bar/baz.pkl'

        with patch.object(obj, 'to_pickle') as mock_to_pickle:
            ballet.util.io._write_tabular_pickle(obj, filepath)

        mock_to_pickle.assert_called_with(filepath)

    def test_write_tabular_pickle_nonarray_raises(self):
        obj = object()
        filepath = '/foo/bar/baz.pkl'
        with self.assertRaises(NotImplementedError):
            ballet.util.io._write_tabular_pickle(obj, filepath)

    def test_write_tabular_h5_ndarray(self):
        obj = self.array
        with tempfile.TemporaryDirectory() as d:
            filepath = pathlib.Path(d).joinpath('baz.h5')
            ballet.util.io._write_tabular_h5(obj, filepath)

            file_size = os.path.getsize(safepath(filepath))
            self.assertGreater(file_size, 0)

    def test_write_tabular_h5_ndframe(self):
        obj = self.frame
        filepath = '/foo/bar/baz.h5'

        with patch.object(obj, 'to_hdf') as mock_to_hdf:
            ballet.util.io._write_tabular_h5(obj, filepath)

        mock_to_hdf.assert_called_with(filepath, key=ANY)

    @unittest.expectedFailure
    def test_read_tabular(self):
        raise NotImplementedError

    @unittest.expectedFailure
    def test_save_model(self):
        raise NotImplementedError

    @unittest.expectedFailure
    def test_save_predictions(self):
        raise NotImplementedError
