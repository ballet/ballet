try:
    import sklearn.impute
    SimpleImputer = sklearn.impute.SimpleImputer
except ImportError:
    import sklearn.preprocessing
    SimpleImputer = sklearn.preprocessing.Imputer
