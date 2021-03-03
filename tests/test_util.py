import sys
import unittest

from ballet.util.code import blacken_code, get_source, is_valid_python
from ballet.util.mod import (  # noqa F401
    import_module_at_path, import_module_from_modname,
    import_module_from_relpath, modname_to_relpath, relpath_to_modname,)


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
