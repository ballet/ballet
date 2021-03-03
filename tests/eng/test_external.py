"""
If the extra is not installed, then the test passes. If the extra is
installed, we should be able to import the last attribute exported from that
module.
"""


def test_category_encoders():
    try:
        import category_encoders  # noqa F401
    except ImportError:
        pass
    else:
        from ballet.eng.category_encoders import WOEEncoder  # noqa F401


def test_feature_engine():
    try:
        import feature_engine  # noqa F401
    except ImportError:
        pass
    else:
        from ballet.eng.feature_engine import (  # noqa f401
            YeoJohnsonTransformer,)


def test_featuretools():
    try:
        import featuretools  # noqa F401
    except ImportError:
        pass
    else:
        from ballet.eng.featuretools import DFSTransformer  # noqa F401


def test_skits():
    try:
        import skits  # noqa F401
    except ImportError:
        pass
    else:
        from ballet.eng.skits import HorizonTransformer  # noqa F401


def test_sklearn():
    from ballet.eng.sklearn import KNNImputer  # noqa F401


def test_tsfresh():
    try:
        import tsfresh  # noqa F401
    except ImportError:
        pass
    else:
        from ballet.eng.tsfresh import FeatureAugmenter  # noqa F401
