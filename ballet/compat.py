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
