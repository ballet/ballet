import ast
import inspect
import platform
from types import FunctionType
from typing import Iterator, Set

import black
from funcy import notnone


def is_valid_python(code: str) -> bool:
    """Check if the string is valid python code

    Args:
        code: code as string
    """
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    else:
        return True


def get_target_python_versions() -> Set[black.TargetVersion]:
    major, minor, _ = platform.python_version_tuple()
    pyversion = f'py{major}{minor}'
    return {black.TargetVersion[pyversion.upper()]}


def blacken_code(code: str) -> str:
    """Format code content using Black

    Args:
        code: code as string
    """
    target_versions = get_target_python_versions()
    line_length = black.DEFAULT_LINE_LENGTH
    string_normalization = True

    mode = black.FileMode(
        target_versions=target_versions,
        line_length=line_length,
        string_normalization=string_normalization,
    )

    try:
        return black.format_file_contents(code, fast=False, mode=mode)
    except black.NothingChanged:
        return code


def get_source(f: FunctionType) -> str:
    """Extract the source code from a given function.

    Recursively extracts the source code for all local functions called by
    given function. The resulting source code is encoded in utf-8.

    Known limitation: Cannot use on function defined interactively in normal
    Python terminal. (Functions defined interactively in IPython are okay.)

    Args:
        f: function

    Returns:
        extracted source code

    Raises:
        NotImplementedError: if the function was defined interactively
    """
    lines = _get_source(f, f.__code__.co_filename, f.__name__, set())
    code = '\n'.join(filter(notnone, lines))

    # post-processing
    if isinstance(code, bytes):
        code = code.decode("utf-8")
    code = blacken_code(code)
    return code


def _get_source(
    f: FunctionType, filename: str, symbolname: str, seen: set
) -> Iterator[str]:
    if f in seen:
        return []

    seen.add(f)

    # known limitation: cannot use from stdin
    if f.__code__.co_filename == '<stdin>':
        raise NotImplementedError(
            'Cannot use \'get_source\' on function defined interactively.')

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
