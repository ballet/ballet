import logging
import os
import pathlib
import random
import sys
import tempfile
import unittest
from unittest.mock import ANY, mock_open, patch

import numpy as np
import pandas as pd
import pytest

import ballet
import ballet.util
import ballet.util.ci
import ballet.util.fs
import ballet.util.io
from ballet.util import one_or_raise
from ballet.util.code import blacken_code, get_source, is_valid_python
from ballet.util.mod import (  # noqa F401
    import_module_at_path, import_module_from_modname,
    import_module_from_relpath, modname_to_relpath, relpath_to_modname,)


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
