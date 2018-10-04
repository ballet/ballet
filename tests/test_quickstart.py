import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from ballet.compat import pathlib
from ballet.util.modutil import import_module_at_path
from ballet.quickstart import generate_project, main

class QuickstartTest(unittest.TestCase):

    @patch('ballet.quickstart.cookiecutter')
    def test_quickstart(self, mock_cookiecutter):
        main()

        args, _ = mock_cookiecutter.call_args

        self.assertEqual(len(args), 1)
        path = args[0]
        self.assertIn('project_template', str(path))


def test_quickstart():
    modname = 'foo'
    extra_context = {
        'project_slug': modname,
    }

    _tempdir = tempfile.TemporaryDirectory()
    tempdir = _tempdir.name

    generate_project(no_input=True, extra_context=extra_context,
                     output_dir=tempdir)

    # make sure we can import different modules without error
    base = pathlib.Path(tempdir).joinpath(modname)

    # 'import foo'
    path_foo = base.joinpath(modname)
    mod_foo = import_module_at_path(modname, path_foo)
    assert isinstance(mod_foo, ModuleType)

    # 'import foo.features'
    path_foo_features = path_foo.joinpath('features')
    mod_foo_features = import_module_at_path(modname + '.' + 'features',
                                             path_foo_features)
    assert isinstance(mod_foo_features, ModuleType)

    _tempdir.cleanup()
