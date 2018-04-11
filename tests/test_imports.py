import unittest


class TestImports(unittest.TestCase):
    def test_feature_imports(self):
        from fhub_core.feature import (  # noqa
            Feature, FeatureValidator, make_robust_transformer,
            RobustTransformerPipeline, make_robust_transformer_pipeline, )

    def test_contrib_imports(self):
        from fhub_core.contrib import (  # noqa
            get_contrib_features, )

    def test_util_imports(self):
        from fhub_core.util import (  # noqa
            asarray2d, get_arr_desc, indent, )

    def test_all_imports(self):
        from fhub_core import (  # noqa
            Feature, FeatureValidator, make_robust_transformer,
            RobustTransformerPipeline, make_robust_transformer_pipeline,
            get_contrib_features, )

    def test_nonexistent_import(self):
        with self.assertRaises(ImportError):
            from fhub_core import bob  # noqa
