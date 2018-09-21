import logging
import pathlib
import tempfile
import unittest
from textwrap import dedent

import funcy

from ballet.contrib import get_contrib_features
from ballet.util.modutil import import_module_at_path

logging.basicConfig(level=logging.DEBUG)


def create_contrib_modules_at_dir(dirname, modcontent, n=1):
    '''Create the a "contrib" directory structure at dirname

    Creates up to mod{n}:

        dirname/
        ├── __init__.py
        ├── mod1/
        |   ├── __init__.py
        |   └── foo1.py
        ├── mod2/
        |   ├── __init__.py
        |   └── foo2.py
        etc.
    '''

    root = pathlib.Path(dirname)
    root.joinpath('__init__.py').touch()
    for i in range(n):
        root.joinpath('mod{i}'.format(i=i)).mkdir()
        root.joinpath('mod{i}'.format(i=i), '__init__.py').touch()

        try:
            modcontent_i = modcontent.format(i=i)
        except KeyError:
            # shouldn't happen
            modcontent_i = modcontent
        modpath_i = str(root.joinpath(
            'mod{i}'.format(i=i), 'foo{i}.py'.format(i=i)))
        with open(modpath_i, 'w') as f:
            f.write(modcontent_i)


class TestContrib(unittest.TestCase):

    def test_get_contrib_features_stdlib(self):
        # give a nonsense *module*, shouldn't import anything. this is a bad
        # test because it relies on module not defining certain names
        import math
        features = get_contrib_features(math)

        # features should be an empty list
        self.assertEqual(len(features), 0)

        # give a nonsense *package*, shouldn't import anything. this is a bad
        # test because it relies on module not defining certain names
        import funcy
        features = get_contrib_features(funcy)
        self.assertEqual(len(features), 0)

    def test_get_contrib_features_generated_modules_components(self):
        n = 4
        content = dedent(
            '''
            from sklearn.preprocessing import StandardScaler
            input = 'col{i}'
            transformer = StandardScaler()
            '''
        )
        modname = 'contrib_features_generated_modules_components'
        with self.mock_contrib_module(modname, content, n) as (mod, features):
            self.assertEqual(len(features), n)

    def test_get_contrib_features_generated_modules_collection(self):
        n = 4
        k = 2
        content = dedent(
            '''
            from ballet import Feature
            from sklearn.preprocessing import StandardScaler
            input = 'col{{i}}'
            transformer = StandardScaler()
            features = [
                Feature(input=input, transformer=transformer),
            ] * {k}
            '''
        ).format(k=k, i='i')
        modname = 'contrib_features_generated_modules_collection'
        with self.mock_contrib_module(modname, content, n) as (mod, features):
            self.assertEqual(len(features), n * k)

    @funcy.contextmanager
    def mock_contrib_module(self, modname, content, n):
        with tempfile.TemporaryDirectory() as tmpdir:
            modpath = pathlib.Path(tmpdir).joinpath(modname)
            modpath.mkdir()
            create_contrib_modules_at_dir(modpath, content, n=n)
            mod = import_module_at_path(modname, modpath)
            features = get_contrib_features(mod)
            yield mod, features
