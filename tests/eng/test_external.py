import unittest


class ExternalTest(unittest.TestCase):

    def test_category_encoders(self):
        try:
            import category_encoders  # noqa F401
        except ImportError:
            pass
        else:
            from ballet.eng.category_encoders import WOEEncoder  # noqa F401

    def test_feature_engine(self):
        try:
            import feature_engine  # noqa F401
        except ImportError:
            pass
        else:
            from ballet.eng.feature_engine import (  # noqa F401
                YeoJohnsonTransformer,)

    def test_featuretools(self):
        try:
            import featuretools  # noqa F401
        except ImportError:
            pass
        else:
            from ballet.eng.featuretools import DFSTransformer  # noqa F401

    def test_skits(self):
        try:
            import skits  # noqa F401
        except ImportError:
            pass
        else:
            from ballet.eng.skits import HorizonTransformer  # noqa F401

    def test_sklearn(self):
        from ballet.eng.sklearn import KNNImputer  # noqa F401

    def test_tsfresh(self):
        try:
            import tsfresh  # noqa F401
        except ImportError:
            pass
        else:
            from ballet.eng.tsfresh import FeatureAugmenter  # noqa F401
