import logging
import os
import pathlib
import random
import sys
import tempfile
import unittest
from unittest.mock import ANY, Mock, mock_open, patch

import git
import numpy as np
import pandas as pd
import pytest
from funcy import identity

import ballet
import ballet.util
import ballet.util.ci
import ballet.util.fs
import ballet.util.git
import ballet.util.io
from ballet.util import one_or_raise
from ballet.util.code import blacken_code, get_source, is_valid_python
from ballet.util.mod import (  # noqa F401
    import_module_at_path, import_module_from_modname,
    import_module_from_relpath, modname_to_relpath, relpath_to_modname,)


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
            assert actual == expected

    def test_replaceext(self):
        filepath0 = '/foo/bar/baz.txt'
        expected = '/foo/bar/baz.py'

        for func in self.fs_path_conversions:
            filepath = func(filepath0)

            new_ext = 'py'
            actual = ballet.util.fs.replaceext(filepath, new_ext)
            assert actual == expected

            new_ext = '.py'
            actual = ballet.util.fs.replaceext(filepath, new_ext)
            assert actual == expected

    def test_splitext2(self):
        filepath0 = '/foo/bar/baz.txt'
        expected = ('/foo/bar', 'baz', '.txt')

        for func in self.fs_path_conversions:
            filepath = func(filepath0)

            actual = ballet.util.fs.splitext2(filepath)
            assert actual == expected

        filepath0 = 'baz.txt'
        expected = ('', 'baz', '.txt')

        for func in self.fs_path_conversions:
            filepath = func(filepath0)

            actual = ballet.util.fs.splitext2(filepath)
            assert actual == expected

    @patch('os.path.exists')
    def test_isemptyfile_does_not_exist(self, mock_exists):
        mock_exists.return_value = False
        result = ballet.util.fs.isemptyfile(
            '/path/to/file/that/hopefully/does/not/exist')
        assert not result

    def test_isemptyfile_is_not_empty(self):
        # file exists and is not empty - false
        with tempfile.TemporaryDirectory() as d:
            filepath = os.path.join(d, 'file')
            with open(filepath, 'w') as f:
                f.write('0')
            result = ballet.util.fs.isemptyfile(filepath)
            assert not result

    def test_isemptyfile_is_empty(self):
        # file exists and is empty - true
        with tempfile.TemporaryDirectory() as d:
            filepath = pathlib.Path(d).joinpath('file')
            filepath.touch()
            result = ballet.util.fs.isemptyfile(filepath)
            assert result

    def test_synctree(self):
        with tempfile.TemporaryDirectory() as tempdir:
            tempdir = pathlib.Path(tempdir).resolve()

            src = tempdir.joinpath('x')
            src.joinpath('a', 'b').mkdir(parents=True)
            src.joinpath('a', 'b', 'only_in_src.txt').touch()
            src.joinpath('a', 'c').mkdir()

            dst = tempdir.joinpath('y')
            dst.joinpath('a', 'b').mkdir(parents=True)
            dst.joinpath('a', 'only_in_dst.txt').touch()

            # patch here in order to avoid messing up tempdir stuff
            with patch('pathlib.Path.mkdir') as mock_mkdir, \
                    patch('os.unlink') as mock_unlink, \
                    patch('os.rmdir') as mock_rmdir, \
                    patch('ballet.util.fs.copyfile') as mock_copyfile:
                ballet.util.fs.synctree(src, dst)

        # one call to mkdir, for 'a/c'
        mock_mkdir.assert_called_once_with()

        # one call to copyfile, for 'only_in_src.txt'
        path = ('a', 'b', 'only_in_src.txt')
        mock_copyfile.assert_called_once_with(
            src.joinpath(*path),
            dst.joinpath(*path)
        )

        # no calls to cleanup
        mock_rmdir.assert_not_called()
        mock_unlink.assert_not_called()

    @unittest.skip('skipping')
    def test__synctree(self):
        # when src is a directory that exists and dst does not exist,
        # then copytree should be called
        src = Mock(spec=pathlib.Path)
        dst = Mock(spec=pathlib.Path)
        dst.exists.return_value = False
        ballet.util.fs._synctree(src, dst, lambda x: None)

    @unittest.expectedFailure
    def test_pwalk(self):
        raise NotImplementedError


class GitTest(unittest.TestCase):

    def test_make_commit_range(self):
        a = 'abc1234'
        b = 'def4321'
        expected_commit_range = 'abc1234...def4321'
        actual_commit_range = ballet.util.git.make_commit_range(a, b)
        assert actual_commit_range == expected_commit_range

    @unittest.expectedFailure
    def test_get_diff_endpoints_from_commit_range(self):
        raise NotImplementedError

    @unittest.expectedFailure
    def test_get_repo(self):
        raise NotImplementedError

    @unittest.expectedFailure
    def test_get_pr_num(self):
        raise NotImplementedError

    @unittest.expectedFailure
    def test_get_branch(self):
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
        ballet.util.git.get_pull_requests(owner, repo, state=state)

        (url, ), kwargs = mock_requests_get.call_args
        assert owner in url
        assert repo in url
        assert 'headers' in kwargs
        assert 'params' in kwargs
        assert kwargs['params']['state'] == state

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
        actual = ballet.util.git.get_pull_request_outcomes(owner, repo)
        assert actual == expected
        mock_get_pull_requests.assert_called_once_with(
            owner, repo, state='closed')

    def test_did_git_push_succeed(self):
        local_ref = None
        remote_ref_string = None
        remote = None

        flags = 0
        push_info = git.remote.PushInfo(flags, local_ref, remote_ref_string,
                                        remote)
        assert ballet.util.git.did_git_push_succeed(push_info)

        flags = git.remote.PushInfo.ERROR
        push_info = git.remote.PushInfo(flags, local_ref, remote_ref_string,
                                        remote)
        assert not ballet.util.git.did_git_push_succeed(push_info)


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
        with pytest.raises(ValueError):
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
        with pytest.raises(NotImplementedError):
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
        with pytest.raises(NotImplementedError):
            ballet.util.io._write_tabular_pickle(obj, filepath)

    def test_write_tabular_h5_ndarray(self):
        obj = self.array
        with tempfile.TemporaryDirectory() as d:
            filepath = pathlib.Path(d).joinpath('baz.h5')
            ballet.util.io._write_tabular_h5(obj, filepath)

            file_size = os.path.getsize(filepath)
            assert file_size > 0

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


class LogTest(unittest.TestCase):

    def setUp(self):
        self.name = str(random.randint(0, 1 << 10))
        self.logger = logging.getLogger(self.name)

    def test_enable(self):
        for level in [logging.INFO, 'CRITICAL']:
            with self.assertLogs(self.logger, level) as cm:
                ballet.util.log.enable(self.logger, level, echo=True)
            msg = one_or_raise(cm.output)
            assert 'enabled' in msg

    def test_level_filter_matches(self):
        ballet.util.log.enable(self.logger, level='DEBUG', echo=False)
        self.logger.addFilter(
            ballet.util.log.LevelFilter(logging.CRITICAL))

        # does log message at level CRITICAL
        with self.assertLogs(self.logger, logging.CRITICAL):
            self.logger.critical('msg')

    def test_level_filter_not_matches(self):
        ballet.util.log.enable(self.logger, level='DEBUG', echo=False)
        self.logger.addFilter(
            ballet.util.log.LevelFilter(logging.DEBUG))

        # does *not* log message at level INFO > DEBUG
        with pytest.raises(AssertionError):
            with self.assertLogs(self.logger, logging.INFO):
                self.logger.info('msg')

    @unittest.expectedFailure
    def test_logging_context(self):
        ballet.util.log.enable(self.logger, level='DEBUG', echo=False)
        with self.assertLogs(self.logger, logging.DEBUG):
            self.logger.debug('msg')

        # TODO not sure why this fails - think the unittest cm is doing
        # something weird
        with ballet.util.log.LoggingContext(self.logger, level='INFO'):
            with pytest.raises(AssertionError):
                with self.assertLogs(self.logger, logging.DEBUG):
                    self.logger.debug('msg')


class CodeTest(unittest.TestCase):

    def test_is_valid_python(self):
        code = '1'
        result = is_valid_python(code)
        assert result

    def test_is_valid_python_invalid(self):
        code = 'this is not valid python code'
        result = is_valid_python(code)
        assert not result

    @unittest.skipUnless(sys.version_info >= (3, 6),
                         "black requires py36 or higher")
    def test_blacken_code(self):
        input = '''\
        x = {  'a':37,'b':42,

        'c':927}
        '''.strip()

        expected = 'x = {"a": 37, "b": 42, "c": 927}'.strip()
        actual = blacken_code(input).strip()

        assert actual == expected

    @unittest.skipUnless(sys.version_info >= (3, 6),
                         "black requires py36 or higher")
    def test_blacken_code_nothing_changed(self):
        input = '1\n'
        expected = '1\n'
        actual = blacken_code(input)

        assert actual == expected

    @unittest.expectedFailure
    def test_get_source(self):
        get_source(None)
