import os


def spliceext(filepath, s):
    """Add s into filepath before the extension"""
    root, ext = os.path.splitext(filepath)
    return root + s + ext


def replaceext(filepath, new_ext):
    """Replace any existing file extension with a new one

    Args:
        filepath (str, path): file path
        new_ext (str): new file extension; if a leading dot is not included,
            it will be added.

    Example::

        >>> replaceext('/foo/bar.txt', 'py')
        '/foo/bar.py'
        >>> replaceext('/foo/bar.txt', '.doc')
        '/foo/bar.doc'
    """
    if new_ext and new_ext[0] != '.':
        new_ext = '.' + new_ext

    root, ext = os.path.splitext(filepath)
    return root + new_ext


def splitext2(filepath):
    """Split filepath into root, filename, ext"""
    root, filename = os.path.split(filepath)
    filename, ext = os.path.splitext(filename)
    return root, filename, ext
