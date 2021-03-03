import logging
import random
import sys
import unittest

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
