import pathlib
import sys
import tempfile
import unittest

from fhub_core.contrib import get_contrib_features


def create_contrib_modules_at_dir(dirname, n=1):
    root = pathlib.Path(dirname)
    root.joinpath('__init__.py').touch()
    root.joinpath('mod1').mkdir()
    root.joinpath('mod1', '__init__.py').touch()
    for i in range(n):
        with open(root.joinpath('mod1', 'foo{}.py'.format(i)), 'w') as f:
            f.write('''
            from sklearn.preprocessing import StandardScaler
            input = 'col1'
            transformer = StandardScaler()
            ''')


def import_module_compat(modname, modpath):
    '''Import module from path

    Source: https://stackoverflow.com/a/67692/2514228
    '''
    modpath = pathlib.Path(modpath)
    if not modpath.parts[-1].endswith('.py'):
        raise ValueError("This won't work")
    if sys.version_info > (3, 5):
        import importlib.util
        spec = importlib.util.spec_from_file_location(modname, modpath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    elif sys.version_info > (3, 3):
        from importlib.machinery import SourceFileLoader
        mod = SourceFileLoader(modname, modpath).load_module()
        return mod
    else:
        raise NotImplementedError


class TestContrib(unittest.TestCase):

    def test_get_contrib_features_stdlib_modules(self):

        # give a nonsense module, shouldn't import anything
        # TODO bad test because relies on module not defining certain names
        import math
        features = get_contrib_features(math)

        # features should be an empty list
        self.assertEqual(len(features), 0)

        # give a nonsense package, shouldn't import anything
        # TODO bad test because relies on module not defining certain names
        import funcy
        features = get_contrib_features(funcy)
        self.assertEqual(len(features), 0)

    @unittest.expectedFailure
    def test_get_contrib_features_generated_modules_components(self):

        with tempfile.TemporaryDirectory() as tmpdirname:
            moddirname = pathlib.Path(tmpdirname) / 'contrib'
            moddirname.mkdir()
            n = 1
            create_contrib_modules_at_dir(moddirname, n=n)

            # TODO this s*** is wh***
            mod = import_module_compat('contrib', moddirname / '__init__.py')
            features = get_contrib_features(mod)

            self.assertEqual(len(features), n)

    @unittest.expectedFailure
    def test_get_contrib_features_generated_modules_collection(self):

        with tempfile.TemporaryDirectory() as tmpdirname:
            moddirname = pathlib.Path(tmpdirname) / 'contrib'
            moddirname.mkdir()
            n = 1
            create_contrib_modules_at_dir(moddirname, n=n)

            # TODO
            raise AssertionError

    @unittest.expectedFailure
    def test_get_contrib_features_generated_modules_mixed(self):

        with tempfile.TemporaryDirectory() as tmpdirname:
            moddirname = pathlib.Path(tmpdirname) / 'contrib'
            moddirname.mkdir()
            n = 1
            create_contrib_modules_at_dir(moddirname, n=n)

            # TODO
            raise AssertionError
