# sklearn compatibility
try:
    import sklearn.impute
    SimpleImputer = sklearn.impute.SimpleImputer
except ImportError:
    import sklearn.preprocessing
    SimpleImputer = sklearn.preprocessing.Imputer

# pathlib compatibility
import sys

if sys.version_info < (3, 5, 0):
    import pathlib2 as pathlib
else:
    import pathlib  # noqa F401

if sys.version_info < (3, 6, 0):
    safepath = str
else:
    from funcy import identity
    safepath = identity

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

