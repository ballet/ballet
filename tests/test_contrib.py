import pathlib
import tempfile
from textwrap import dedent

from funcy import contextmanager

from ballet.contrib import _collect_contrib_features
from ballet.util.mod import import_module_at_path


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
        root.joinpath(f'mod{i}').mkdir()
        root.joinpath(f'mod{i}', '__init__.py').touch()

        try:
            modcontent_i = modcontent.format(i=i)
        except KeyError:
            # shouldn't happen
            modcontent_i = modcontent
        modpath_i = str(root.joinpath(f'mod{i}', f'foo{i}.py'))
        with open(modpath_i, 'w') as f:
            f.write(modcontent_i)


@contextmanager
def mock_contrib_module(modname, content, n):
    with tempfile.TemporaryDirectory() as tmpdir:
        modpath = pathlib.Path(tmpdir).joinpath(modname)
        modpath.mkdir()
        create_contrib_modules_at_dir(modpath, content, n=n)
        mod = import_module_at_path(modname, modpath)
        features = _collect_contrib_features(mod)
        yield mod, features


def test_collect_contrib_features_stdlib():
    # give a nonsense *module*, shouldn't import anything. this is a bad
    # test because it relies on module not defining certain names
    import math
    features = _collect_contrib_features(math)

    # features should be an empty list
    assert len(features) == 0


def test_collect_contrib_features_thirdparty():
    # give a nonsense *package*, shouldn't import anything. this is a bad
    # test because it relies on module not defining certain names
    import funcy
    features = _collect_contrib_features(funcy)
    assert len(features) == 0


def test_collect_contrib_features_generated():
    n = 4
    content = dedent(
        '''
        from ballet import Feature
        from sklearn.preprocessing import StandardScaler
        input = 'col{i}'
        transformer = StandardScaler()
        feature = Feature(input, transformer)
        '''
    ).strip()
    modname = 'contrib_features_generated'
    with mock_contrib_module(modname, content, n) as (mod, features):
        assert len(features) == n
