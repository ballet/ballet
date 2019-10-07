import inspect
import platform

from funcy import notnone

from ballet.compat import black


def blacken_code(code):
    """Format code content using Black

    Args:
        code (str): code as string

    Returns:
        str
    """
    if black is None:
        raise NotImplementedError

    major, minor, _ = platform.python_version_tuple()
    pyversion = 'py{major}{minor}'.format(major=major, minor=minor)
    target_versions = [black.TargetVersion[pyversion.upper()]]

    line_length = black.DEFAULT_LINE_LENGTH
    string_normalization = True

    mode = black.FileMode(
        target_versions=target_versions,
        line_length=line_length,
        string_normalization=string_normalization,
    )

    return black.format_file_contents(code, fast=False, mode=mode)


def get_source(f):
    """Extract the source code from a given function.

    Recursively extracts the source code for all local functions called by
    given function. The resulting source code is encoded in utf-8.

    Known limitation: Cannot use on function defined interactively in normal
    Python terminal. (Functions defined interactively in IPython are okay.)

    Args:
        f: function

    Returns:
        str: extracted source code

    Raises:
        NotImplementedError: if the function was defined interactively
    """
    code = _get_source(f, f.__code__.co_filename, f.__name__, set())
    code = filter(notnone, code)
    code = '\n'.join(code)

    # post-processing
    if isinstance(code, bytes):
        code = code.decode("utf-8")
    code = blacken_code(code)
    return code


def _get_source(f, filename, symbolname, seen):
    if f in seen:
        return []

    seen.add(f)

    # known limitation: cannot use from stdin
    if f.__code__.co_filename == '<stdin>':
        raise NotImplementedError(
            'Cannot use {name!r} on function defined interactively.'
            .format(name='get_source'))

    # if f was not defined in the same file, return code to import it
    # TODO
    if f.__code__.co_filename != filename:
        return []

    # get source of referenced symbols
    for symbolname in f.__code__.co_names:
        obj = f.__globals__.get(symbolname)
        if obj and inspect.isfunction(obj):
            yield from _get_source(obj, filename, symbolname, seen)

    # get source of self
    yield inspect.getsource(f)
