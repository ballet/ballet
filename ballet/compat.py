# sklearn compatibility
try:
    import sklearn.impute
    SimpleImputer = sklearn.impute.SimpleImputer
except ImportError:
    import sklearn.preprocessing
    SimpleImputer = sklearn.preprocessing.Imputer

# pathlib compatibility
import sys
import contextlib

redirect_stdout = contextlib.redirect_stdout
if sys.version_info < (3, 5, 0):
    import pathlib2 as pathlib
    def redirect_stderr(f):
        sys.stderr = f
else:
    import pathlib  # noqa F401
    redirect_stderr = contextlib.redirect_stderr
if sys.version_info < (3, 6, 0):
    safepath = str
else:
    from funcy import identity
    safepath = identity
