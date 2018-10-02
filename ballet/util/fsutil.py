import os


def spliceext(filepath, s):
    """Add s into filepath before the extension"""
    root, ext = os.path.splitext(filepath)
    return root + s + ext


def replaceext(filepath, new_ext):
    """Replace any existing file extension with a new one"""
    root, ext = os.path.splitext(filepath)
    return root + new_ext


def splitext2(filepath):
    """Split filepath into root, filename, ext"""
    root, filename = os.path.split(filepath)
    filename, ext = os.path.splitext(filename)
    return root, filename, ext
