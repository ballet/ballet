import logging
import pathlib
import pkgutil
import tempfile
import unittest
from textwrap import dedent

from fhub_core.contrib import get_contrib_features

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
        with open(root.joinpath('mod{i}'.format(i=i),
                                'foo{i}.py'.format(i=i)), 'w') as f:
            f.write(modcontent_i)


def import_module_at_path(modname, modpath):
    '''Import module from path'''
    modpath = pathlib.Path(modpath)
    parentpath = str(modpath.parent)
    modpath = str(modpath)
    importer = pkgutil.get_importer(parentpath)
    mod = importer.find_module(modname).load_module(modname)
    return mod


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
        modcontent = dedent(
            '''\
            from sklearn.preprocessing import StandardScaler
            input = 'col{i}'
            transformer = StandardScaler()
            ''')
        with tempfile.TemporaryDirectory() as tmpdirname:
            moddirname = pathlib.Path(tmpdirname) / 'contrib_components'
            moddirname.mkdir()
            create_contrib_modules_at_dir(moddirname, modcontent, n=n)
            mod = import_module_at_path('contrib_components', moddirname)
            features = get_contrib_features(mod)

        self.assertEqual(len(features), n)

    def test_get_contrib_features_generated_modules_collection(self):
        n = 4
        k = 2
        f = 'Feature(input=input, transformer=transformer),\n' * k
        modcontent = dedent(
            '''\
            from fhub_core import Feature
            from sklearn.preprocessing import StandardScaler
            input = 'col{{i}}'
            transformer = StandardScaler()
            features = [
                {f}
            ]
            ''').format(f=f, i='i')
        with tempfile.TemporaryDirectory() as tmpdirname:
            moddirname = pathlib.Path(tmpdirname) / 'contrib_collection'
            moddirname.mkdir()
            create_contrib_modules_at_dir(moddirname, modcontent, n=n)
            mod = import_module_at_path('contrib_collection', moddirname)
            features = get_contrib_features(mod)

        self.assertEqual(len(features), n * k)
