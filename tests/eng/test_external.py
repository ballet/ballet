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
        from ballet.eng.external.category_encoders import (  # noqa F401
            WOEEncoder,)


def test_feature_engine():
    try:
        import feature_engine  # noqa F401
    except ImportError:
        pass
    else:
        from ballet.eng.external.feature_engine import (  # noqa f401
            YeoJohnsonTransformer,)


def test_featuretools():
    try:
        import featuretools  # noqa F401
    except ImportError:
        pass
    else:
        from ballet.eng.external.featuretools import (  # noqa F401
            DFSTransformer,)


def test_skits():
    try:
        import skits  # noqa F401
    except ImportError:
        pass
    else:
        from ballet.eng.external.skits import HorizonTransformer  # noqa F401


def test_sklearn():
    from ballet.eng.external.sklearn import KNNImputer  # noqa F401


def test_tsfresh():
    try:
        import tsfresh  # noqa F401
    except ImportError:
        pass
    else:
        from ballet.eng.external.tsfresh import FeatureAugmenter  # noqa F401
