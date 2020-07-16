import sys


# sklearn compatibility
try:
    from sklearn.impute import SimpleImputer
except ImportError:
    import sklearn.preprocessing
    SimpleImputer = sklearn.preprocessing.Imputer


if sys.version_info < (3, 6):
    safepath = str
else:
    from funcy import identity as _identity
    safepath = _identity


try:
    from os import PathLike
except ImportError:
    from pathlib import Path
    PathLike = (Path, )


# redirect_stderr new in 3.5
from contextlib import redirect_stdout  # noqa F401
try:
    from contextlib import redirect_stderr
except ImportError:
    from contextlib import contextmanager
    @contextmanager
    def redirect_stderr(f):
        oldstream = sys.stderr
        try:
            sys.stderr = f
            yield f
        finally:
            sys.stderr = oldstream


# black compatibility - only installable for python 3.6+
try:
    import black
except ImportError:
    black = None


# nullcontext new in 3.7?
try:
    from contextlib import nullcontext
except ImportError:
    from funcy import nullcontext


if sys.version_info >= (3, 7):
    import pathlib as _pathlib
    is_mount = _pathlib.Path.is_mount
else:
    # condensed python 3.8 pathlib source
    def is_mount(path):
        from pathlib import Path
        if not path.exists() or not path.is_dir():
            return False
        parent = Path(path.parent)
        try:
            parent_dev = parent.stat().st_dev
        except OSError:
            return False
        dev = path.stat().st_dev
        if dev != parent_dev:
            return True
        ino = path.stat().st_ino
        parent_ino = parent.stat().st_ino
        return ino == parent_ino
