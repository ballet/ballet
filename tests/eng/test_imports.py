import unittest


class TestImports(unittest.TestCase):
    def test_ts_imports(self):
        from ballet.eng.ts import (  # noqa
            SingleLagger, make_multi_lagger, )

    def test_missing_imports(self):
        from ballet.eng.missing import (  # noqa
            LagImputer, NullFiller, NullIndicator, )

    def test_base_imports(self):
        from ballet.eng.base import (  # noqa
            NoFitMixin, SimpleFunctionTransformer,
            GroupedFunctionTransformer, )

    def test_misc_imports(self):
        from ballet.eng.misc import (  # noqa
            IdentityTransformer, ValueReplacer, NamedFramer, )

    def test_all_imports(self):
        from ballet.eng import (  # noqa
            SingleLagger, make_multi_lagger,
            LagImputer, NullFiller, NullIndicator,
            NoFitMixin, SimpleFunctionTransformer, GroupedFunctionTransformer,
            IdentityTransformer, ValueReplacer, NamedFramer, )

    def test_nonexistent_import(self):
        with self.assertRaises(ImportError):
            from ballet.eng import bob  # noqa
