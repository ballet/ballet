import os
import os.path
import pathlib
from shutil import copyfile

from funcy import partial, suppress

from ballet.compat import safepath
from ballet.exc import BalletError
from ballet.util.log import logger


def spliceext(filepath, s):
    """Add s into filepath before the extension

    Args:
        filepath (PathLike): file path
        s (str): string to splice

    Returns:
        str
    """
    root, ext = os.path.splitext(safepath(filepath))
    return root + s + ext


def replaceext(filepath, new_ext):
    """Replace any existing file extension with a new one

    If the new extension is the empty string, all existing extensions will
    be removed.

    Example::

        >>> replaceext('/foo/bar.txt', 'py')
        '/foo/bar.py'
        >>> replaceext('/foo/bar.txt', '.doc')
        '/foo/bar.doc'

    Args:
        filepath (PathLike): file path
        new_ext (str): new file extension; if a leading dot is not included,
            it will be added.

    Returns:
        str
    """
    if new_ext and not new_ext.startswith('.'):
        new_ext = '.' + new_ext
    return str(pathlib.Path(filepath).with_suffix(new_ext))


def splitext2(filepath):
    """Split filepath into root, filename, ext

    Args:
        filepath (PathLike): file path

    Returns:
        str
    """
    root, filename = os.path.split(safepath(filepath))
    filename, ext = os.path.splitext(safepath(filename))
    return root, filename, ext


def isemptyfile(filepath):
    """Determine if the file both exists and isempty

    Args:
        filepath (PathLike): file path

    Returns:
        bool
    """
    exists = os.path.exists(safepath(filepath))
    if exists:
        filesize = os.path.getsize(safepath(filepath))
        return filesize == 0
    else:
        return False


def synctree(src, dst, onexist=None):
    """Recursively sync files at directory src to dst

    This is more or less equivalent to::

       cp -n -R ${src}/ ${dst}/

    If a file at the same path exists in src and dst, it is NOT overwritten
    in dst. Pass ``onexist`` in order to raise an error on such conditions.

    Args:
        src (path-like): source directory
        dst (path-like): destination directory, does not need to exist
        onexist (callable): function to call if file exists at destination,
            takes the full path to destination file as only argument

    Returns:
        List[Tuple[PathLike,str]]: changes made by synctree, list of tuples of
            the form ("/absolute/path/to/file", "<kind>") where the change
            kind is one of "dir" (new directory was created) or "file" (new
            file was created).
    """
    src = pathlib.Path(src).resolve()
    dst = pathlib.Path(dst).resolve()

    if not src.is_dir():
        raise ValueError

    if dst.exists() and not dst.is_dir():
        raise ValueError

    if onexist is None:
        def onexist(): pass

    return _synctree(src, dst, onexist)


def _synctree(src, dst, onexist):
    result = []
    cleanup = []
    try:
        for root, dirnames, filenames in os.walk(safepath(src)):
            root = pathlib.Path(root)
            relative_dir = root.relative_to(src)

            for dirname in dirnames:
                dstdir = dst.joinpath(relative_dir, dirname)
                if dstdir.exists():
                    if not dstdir.is_dir():
                        raise BalletError
                else:
                    logger.debug(
                        'Making directory: {dstdir!s}'.format(dstdir=dstdir))
                    dstdir.mkdir()
                    result.append((dstdir, 'dir'))
                    cleanup.append(partial(os.rmdir, safepath(dstdir)))

            for filename in filenames:
                srcfile = root.joinpath(filename)
                dstfile = dst.joinpath(relative_dir, filename)
                if dstfile.exists():
                    onexist(dstfile)
                else:
                    logger.debug(
                        'Copying file to destination: {dstfile!s}'
                        .format(dstfile=dstfile))
                    copyfile(srcfile, dstfile)
                    result.append((dstfile, 'file'))
                    cleanup.append(partial(os.unlink, safepath(dstfile)))

    except Exception:
        with suppress(Exception):
            for f in reversed(cleanup):
                f()
        raise

    return result


def pwalk(d, **kwargs):
    """Similar to os.walk but with pathlib.Path objects

    Returns:
        Iterable[Path]
    """
    for dirpath, dirnames, filenames in os.walk(safepath(d), **kwargs):
        dirpath = pathlib.Path(dirpath)
        for p in dirnames + filenames:
            yield dirpath.joinpath(p)
