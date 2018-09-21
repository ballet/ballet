import unittest


class TestImports(unittest.TestCase):
    def test_feature_imports(self):
        from ballet.feature import (  # noqa
            Feature, make_robust_transformer,
            RobustTransformerPipeline, make_robust_transformer_pipeline, )

    def test_contrib_imports(self):
        from ballet.contrib import (  # noqa
            get_contrib_features, )

    def test_util_imports(self):
        from ballet.util import (  # noqa
            asarray2d, get_arr_desc, indent, )

        from ballet.util.gitutil import (  # noqa
            PullRequestInfo, HeadInfo, get_diffs_by_revision,
            get_diffs_by_diff_str,
            PullRequestBuildDiffer)

        from ballet.util.modutil import (  # noqa
            import_module_at_path, import_module_from_modname,
            import_module_from_relpath, modname_to_relpath,
            relpath_to_modname, )

        from ballet.util.travisutil import (  # noqa
            get_travis_pr_num, is_travis_pr,
            TravisPullRequestBuildDiffer)

    def test_validation_imports(self):
        from ballet.validation import (  # noqa
            FeatureValidator, PullRequestFeatureValidator, )

    def test_toplevel_imports(self):
        from ballet import (  # noqa
            Feature, make_robust_transformer,
            RobustTransformerPipeline, make_robust_transformer_pipeline,
            get_contrib_features, )

    def test_nonexistent_import(self):
        with self.assertRaises(ImportError):
            from ballet import bob  # noqa
