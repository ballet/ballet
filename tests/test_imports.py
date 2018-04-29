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

        from fhub_core.util.travis import (  # noqa
            get_travis_pr_num, is_travis_pr, )

        from fhub_core.util.modutil import (  # noqa
            import_module_at_path, import_module_from_modname,
            import_module_from_relpath, modname_to_relpath,
            relpath_to_modname, )

        from fhub_core.util.git import (  # noqa
            PullRequestInfo, HeadInfo, get_file_changes_by_revision,
            get_file_changes_by_diff_str, )

    def test_validation_imports(self):
        from fhub_core.validation import (  # noqa
            PullRequestFeatureValidator, )

    def test_toplevel_imports(self):
        from fhub_core import (  # noqa
            Feature, FeatureValidator, make_robust_transformer,
            RobustTransformerPipeline, make_robust_transformer_pipeline,
            get_contrib_features, )

    def test_nonexistent_import(self):
        with self.assertRaises(ImportError):
            from fhub_core import bob  # noqa
