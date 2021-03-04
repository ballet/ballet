import os
import os.path
import pathlib
from shutil import copyfile
from typing import Callable, Iterator, List, Tuple

from funcy import partial, suppress

from ballet.exc import BalletError
from ballet.util.log import logger
from ballet.util.typing import Pathy


def spliceext(filepath: Pathy, s: str) -> str:
    """Add s into filepath before the extension"""
    root, ext = os.path.splitext(filepath)
    return root + s + ext


def replaceext(filepath: Pathy, new_ext: str) -> str:
    """Replace any existing file extension with a new one

    If the new extension is the empty string, all existing extensions will
    be removed.

    Example::

        >>> replaceext('/foo/bar.txt', 'py')
        '/foo/bar.py'
        >>> replaceext('/foo/bar.txt', '.doc')
        '/foo/bar.doc'

    Args:
        filepath: file path
        new_ext: new file extension; if a leading dot is not included, it will
            be added.
    """
    if new_ext and not new_ext.startswith('.'):
        new_ext = '.' + new_ext
    return str(pathlib.Path(filepath).with_suffix(new_ext))


def splitext2(filepath: Pathy) -> Tuple[str, str, str]:
    """Split filepath into root, filename, ext"""
    root, filename = os.path.split(filepath)
    filename, ext = os.path.splitext(filename)
    return root, filename, ext


def isemptyfile(filepath: Pathy) -> bool:
    """Determine if the file both exists and isempty"""
    exists = os.path.exists(filepath)
    if exists:
        filesize = os.path.getsize(filepath)
        return filesize == 0
    else:
        return False


def synctree(
    src: Pathy,
    dst: Pathy,
    onexist: Callable[[pathlib.Path], None] = None
) -> List[Tuple[pathlib.Path, str]]:
    """Recursively sync files at directory src to dst

    This is more or less equivalent to::

       cp -n -R ${src}/ ${dst}/

    If a file at the same path exists in src and dst, it is NOT overwritten
    in dst. Pass ``onexist`` in order to raise an error on such conditions.

    Args:
        src: source directory
        dst: destination directory, does not need to exist
        onexist: function to call if file exists at destination,
            takes the full path to destination file as only argument

    Returns:
        changes made by synctree, list of tuples of the form
        ("/absolute/path/to/file", "<kind>") where the change kind is one of
        "dir" (new directory was created) or "file" (new file was created).
    """
    src = pathlib.Path(src).resolve()
    dst = pathlib.Path(dst).resolve()

    if not src.is_dir():
        raise ValueError

    if dst.exists() and not dst.is_dir():
        raise ValueError

    if onexist is None:
        def _onexist(path): pass
        onexist = _onexist

    return _synctree(src, dst, onexist)


def _synctree(
    src: pathlib.Path,
    dst: pathlib.Path,
    onexist: Callable[[pathlib.Path], None]
) -> List[Tuple[pathlib.Path, str]]:
    result = []
    cleanup = []
    try:
        for _root, dirnames, filenames in os.walk(src):
            root = pathlib.Path(_root)
            relative_dir = root.relative_to(src)

            for dirname in dirnames:
                dstdir = dst.joinpath(relative_dir, dirname)
                if dstdir.exists():
                    if not dstdir.is_dir():
                        raise BalletError
                else:
                    logger.debug(f'Making directory: {dstdir!s}')
                    dstdir.mkdir()
                    result.append((dstdir, 'dir'))
                    cleanup.append(partial(os.rmdir, dstdir))

            for filename in filenames:
                srcfile = root.joinpath(filename)
                dstfile = dst.joinpath(relative_dir, filename)
                if dstfile.exists():
                    onexist(dstfile)
                else:
                    logger.debug(f'Copying file to destination: {dstfile!s}')
                    copyfile(srcfile, dstfile)
                    result.append((dstfile, 'file'))
                    cleanup.append(partial(os.unlink, dstfile))

    except Exception:
        with suppress(Exception):
            for f in reversed(cleanup):
                f()
        raise

    return result


def pwalk(d: Pathy, **kwargs) -> Iterator[pathlib.Path]:
    """Similar to os.walk but with pathlib.Path objects"""
    for _dirpath, dirnames, filenames in os.walk(d, **kwargs):
        dirpath = pathlib.Path(_dirpath)
        for p in dirnames + filenames:
            yield dirpath.joinpath(p)
