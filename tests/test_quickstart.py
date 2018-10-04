import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from ballet.compat import pathlib
from ballet.eng.misc import IdentityTransformer
from ballet.feature import Feature
from ballet.util.modutil import import_module_at_path, modname_to_relpath
from ballet.quickstart import generate_project, main
from sklearn_pandas import DataFrameMapper

class QuickstartTest(unittest.TestCase):

    @patch('ballet.quickstart.cookiecutter')
    def test_quickstart(self, mock_cookiecutter):
        main()

        args, _ = mock_cookiecutter.call_args

        self.assertEqual(len(args), 1)
        path = args[0]
        self.assertIn('project_template', str(path))


def test_quickstart():
    modname = 'foo'
    extra_context = {
        'project_slug': modname,
    }

    _tempdir = tempfile.TemporaryDirectory()
    tempdir = _tempdir.name

    generate_project(no_input=True, extra_context=extra_context,
                     output_dir=tempdir)

    # make sure we can import different modules without error
    base = pathlib.Path(tempdir).joinpath(modname)

    def _import(modname):
        relpath = modname_to_relpath(modname)
        abspath = base.joinpath(relpath)
        return import_module_at_path(modname, abspath)

    foo = _import('foo')
    assert isinstance(foo, ModuleType)

    foo_features = _import('foo.features')
    assert isinstance(foo_features, ModuleType)

    foo_features_buildfeatures = _import('foo.features.build_features')
    assert isinstance(foo_features_buildfeatures, ModuleType)

    get_contrib_features = foo_features_buildfeatures.get_contrib_features
    features = get_contrib_features()
    assert len(features) == 0

    # first providing a mock feature, call build_features
    with patch.object(
        foo_features_buildfeatures, 'get_contrib_features',
        return_value=[Feature(input='A', transformer=IdentityTransformer)]
    ):
        X_df = pd.util.testing.makeCustomDataframe(5, 2)
        X_df.columns = ['A', 'B']
        X, mapper = foo_features_buildfeatures.build_features(X_df)
        assert np.shape(X) == (5, 1)
        assert isinstance(mapper, DataFrameMapper)

    _tempdir.cleanup()
