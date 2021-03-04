import pathlib
import types

import pytest

import ballet
from ballet.util.mod import (
    import_module_at_path, modname_to_relpath, relpath_to_modname,)


@pytest.mark.xfail
def test_import_module_from_modname():
    import ballet.util.mod.import_module_from_modname  # noqa F401
    raise NotImplementedError


@pytest.mark.xfail
def test_import_module_from_relpath():
    import ballet.util.mod.import_module_from_relpath  # noqa F401
    raise NotImplementedError


def test_import_module_at_path_module(tmp_path):
    path = tmp_path.joinpath('foo', 'bar.py')
    path.parent.mkdir(parents=True)
    init = path.parent.joinpath('__init__.py')
    init.touch()
    x = 1
    with path.open('w') as f:
        f.write(f'x={x!r}')
    modname = 'foo.bar'
    modpath = str(path)  # e.g. /tmp/foo/bar.py'
    mod = import_module_at_path(modname, modpath)
    assert isinstance(mod, types.ModuleType)
    assert mod.__name__ == modname
    assert mod.x == x


def test_import_module_at_path_package(tmp_path):
    path = tmp_path.joinpath('foo')
    path.mkdir(parents=True)
    init = path.joinpath('__init__.py')
    init.touch()
    x = 'hello'
    with init.open('w') as f:
        f.write(f'x={x!r}')
    modname = 'foo'
    modpath = str(path)
    mod = import_module_at_path(modname, modpath)
    assert isinstance(mod, types.ModuleType)
    assert mod.__name__ == modname
    assert mod.x == x


@pytest.mark.xfail
def test_import_module_at_path_bad_package_structure():
    raise NotImplementedError


def test_relpath_to_modname():
    relpath = 'ballet/util/_util.py'
    expected_modname = 'ballet.util._util'
    actual_modname = relpath_to_modname(relpath)
    assert actual_modname == expected_modname

    relpath = 'ballet/util/__init__.py'
    expected_modname = 'ballet.util'
    actual_modname = relpath_to_modname(relpath)
    assert actual_modname == expected_modname

    relpath = 'ballet/foo/bar/baz.zip'
    with pytest.raises(ValueError):
        relpath_to_modname(relpath)


def test_modname_to_relpath_module():
    modname = 'ballet.util._util'
    expected_relpath = 'ballet/util/_util.py'
    actual_relpath = modname_to_relpath(modname)
    assert actual_relpath == expected_relpath


def test_modname_to_relpath_package():
    modname = 'ballet.util'
    # mypackage.__file__ resolves to the __init__.py
    project_root = pathlib.Path(ballet.__file__).parent.parent

    expected_relpath = 'ballet/util/__init__.py'
    actual_relpath = modname_to_relpath(modname, project_root=project_root)
    assert actual_relpath == expected_relpath

    # TODO patch this
    # # without providing project root, behavior is undefined, as we don't
    # # know whether the relative path will resolve to a directory

    # # within a temporary directory, the relpath *should not* be a dir
    # with tempfile.TemporaryDirectory() as tmpdir:
    #     cwd = os.getcwd()
    #     try:
    #         os.chdir(tmpdir)
    #         actual_relpath = modname_to_relpath(modname)
    #         expected_relpath = 'ballet/util.py'
    #         self.assertEqual(actual_relpath, expected_relpath)
    #     finally:
    #         os.chdir(cwd)

    # # from the actual project root, the relpath *should* be a dir
    # cwd = os.getcwd()
    # try:
    #     os.chdir(str(project_root))
    #     actual_relpath = modname_to_relpath(modname)
    #     expected_relpath = 'ballet/util/__init__.py'
    #     self.assertEqual(actual_relpath, expected_relpath)
    # finally:
    #     os.chdir(cwd)


def test_modname_to_relpath_package_no_add_init():
    project_root = pathlib.Path(ballet.__file__).parent.parent
    # without add_init
    modname = 'ballet.util'
    add_init = False
    expected_relpath = 'ballet/util'
    actual_relpath = modname_to_relpath(
        modname, project_root=project_root, add_init=add_init)
    assert actual_relpath == expected_relpath
