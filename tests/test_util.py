import os
import pathlib
import tempfile
import unittest

import fhub_core
from fhub_core.util.modutil import (  # noqa F401
    import_module_at_path, import_module_from_modname,
    import_module_from_relpath, modname_to_relpath, relpath_to_modname)


class TestModutil(unittest.TestCase):

    @unittest.expectedFailure
    def test_import_module_from_modname(self):
        raise NotImplementedError

    @unittest.expectedFailure
    def test_import_module_from_relpath(self):
        raise NotImplementedError

    @unittest.expectedFailure
    def test_import_module_at_path(self):
        raise NotImplementedError

    def test_relpath_to_modname(self):
        relpath = 'fhub_core/util/_util.py'
        expected_modname = 'fhub_core.util._util'
        actual_modname = relpath_to_modname(relpath)
        self.assertEqual(actual_modname, expected_modname)

        relpath = 'fhub_core/util/__init__.py'
        expected_modname = 'fhub_core.util'
        actual_modname = relpath_to_modname(relpath)
        self.assertEqual(actual_modname, expected_modname)

        relpath = 'fhub_core/foo/bar/baz.zip'
        with self.assertRaises(ValueError):
            relpath_to_modname(relpath)

    def test_modname_to_relpath(self):
        modname = 'fhub_core.util._util'
        expected_relpath = 'fhub_core/util/_util.py'
        actual_relpath = modname_to_relpath(modname)
        self.assertEqual(actual_relpath, expected_relpath)

        modname = 'fhub_core.util'
        # mypackage.__file__ resolves to the __init__.py
        project_root = pathlib.Path(fhub_core.__file__).parent.parent

        expected_relpath = 'fhub_core/util/__init__.py'
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
                expected_relpath = 'fhub_core/util.py'
                self.assertEqual(actual_relpath, expected_relpath)
            finally:
                os.chdir(cwd)

        # from the actual project root, the relpath *should* be a dir
        cwd = os.getcwd()
        try:
            os.chdir(str(project_root))
            actual_relpath = modname_to_relpath(modname)
            expected_relpath = 'fhub_core/util/__init__.py'
            self.assertEqual(actual_relpath, expected_relpath)
        finally:
            os.chdir(cwd)

        # without add_init
        modname = 'fhub_core.util'
        add_init = False
        expected_relpath = 'fhub_core/util'
        actual_relpath = modname_to_relpath(
            modname, project_root=project_root, add_init=add_init)
        self.assertEqual(actual_relpath, expected_relpath)
