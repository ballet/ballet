import os.path

from ballet.compat import safepath


def spliceext(filepath, s):
    """Add s into filepath before the extension

    Args:
        filepath (str, path): file path
        s (str): string to splice

    Returns:
        str
    """
    root, ext = os.path.splitext(safepath(filepath))
    return root + s + ext


def replaceext(filepath, new_ext):
    """Replace any existing file extension with a new one

    Example::

        >>> replaceext('/foo/bar.txt', 'py')
        '/foo/bar.py'
        >>> replaceext('/foo/bar.txt', '.doc')
        '/foo/bar.doc'

    Args:
        filepath (str, path): file path
        new_ext (str): new file extension; if a leading dot is not included,
            it will be added.

    Returns:
        Tuple[str]
    """
    if new_ext and new_ext[0] != '.':
        new_ext = '.' + new_ext

    root, ext = os.path.splitext(safepath(filepath))
    return root + new_ext


def splitext2(filepath):
    """Split filepath into root, filename, ext

    Args:
        filepath (str, path): file path

    Returns:
        str
    """
    root, filename = os.path.split(safepath(filepath))
    filename, ext = os.path.splitext(safepath(filename))
    return root, filename, ext


def isemptyfile(filepath):
    """Determine if the file both exists and isempty

    Args:
        filepath (str, path): file path

    Returns:
        bool
    """
    exists = os.path.exists(safepath(filepath))
    if exists:
        filesize = os.path.getsize(safepath(filepath))
        return filesize == 0
    else:
        return False
