import pathlib
from unittest.mock import Mock, patch

import pytest
from funcy import identity

from ballet.util.fs import (
    _synctree, isemptyfile, replaceext, spliceext, splitext2, synctree,)


@pytest.mark.parametrize(
    'convert_path',
    [identity, pathlib.Path],
)
def test_spliceext(convert_path):
    filepath0 = '/foo/bar/baz.txt'

    filepath = convert_path(filepath0)
    s = '_new'
    expected = '/foo/bar/baz_new.txt'

    actual = spliceext(filepath, s)
    assert actual == expected


@pytest.mark.parametrize(
    'convert_path',
    [identity, pathlib.Path],
)
def test_replaceext(convert_path):
    filepath0 = '/foo/bar/baz.txt'
    expected = '/foo/bar/baz.py'

    filepath = convert_path(filepath0)

    new_ext = 'py'
    actual = replaceext(filepath, new_ext)
    assert actual == expected

    new_ext = '.py'
    actual = replaceext(filepath, new_ext)
    assert actual == expected


@pytest.mark.parametrize(
    'convert_path',
    [identity, pathlib.Path],
)
def test_splitext2(convert_path):
    filepath0 = '/foo/bar/baz.txt'
    expected = ('/foo/bar', 'baz', '.txt')

    filepath = convert_path(filepath0)

    actual = splitext2(filepath)
    assert actual == expected

    filepath0 = 'baz.txt'
    expected = ('', 'baz', '.txt')

    filepath = convert_path(filepath0)

    actual = splitext2(filepath)
    assert actual == expected


@patch('os.path.exists')
def test_isemptyfile_does_not_exist(mock_exists):
    mock_exists.return_value = False
    result = isemptyfile(
        '/path/to/file/that/hopefully/does/not/exist')
    assert not result


def test_isemptyfile_is_not_empty(tmp_path):
    # file exists and is not empty - false
    filepath = tmp_path.joinpath('file')
    with filepath.open('w') as f:
        f.write('0')
    result = isemptyfile(filepath)
    assert not result


def test_isemptyfile_is_empty(tmp_path):
    # file exists and is empty - true
    filepath = tmp_path.joinpath('file')
    filepath.touch()
    result = isemptyfile(filepath)
    assert result


def test_synctree(tmp_path):
    src = tmp_path.joinpath('x')
    src.joinpath('a', 'b').mkdir(parents=True)
    src.joinpath('a', 'b', 'only_in_src.txt').touch()
    src.joinpath('a', 'c').mkdir()

    dst = tmp_path.joinpath('y')
    dst.joinpath('a', 'b').mkdir(parents=True)
    dst.joinpath('a', 'only_in_dst.txt').touch()

    # patch here in order to avoid messing up tmp_path stuff
    with patch('pathlib.Path.mkdir') as mock_mkdir, \
            patch('os.unlink') as mock_unlink, \
            patch('os.rmdir') as mock_rmdir, \
            patch('ballet.util.fs.copyfile') as mock_copyfile:
        synctree(src, dst)

    # one call to mkdir, for 'a/c'
    mock_mkdir.assert_called_once_with()

    # one call to copyfile, for 'only_in_src.txt'
    path = ('a', 'b', 'only_in_src.txt')
    mock_copyfile.assert_called_once_with(
        src.joinpath(*path),
        dst.joinpath(*path)
    )

    # no calls to cleanup
    mock_rmdir.assert_not_called()
    mock_unlink.assert_not_called()


@pytest.mark.skip(reason='skipping')
def test__synctree():
    # when src is a directory that exists and dst does not exist,
    # then copytree should be called
    src = Mock(spec=pathlib.Path)
    dst = Mock(spec=pathlib.Path)
    dst.exists.return_value = False
    _synctree(src, dst, lambda x: None)


@pytest.mark.xfail
def test_pwalk():
    raise NotImplementedError
