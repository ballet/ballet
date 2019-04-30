import platform

import black


def blacken_code(code):
    """Format code content using Black

    Args:
        code (str): code as string

    Returns:
        str
    """
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
