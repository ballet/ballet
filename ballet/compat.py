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
