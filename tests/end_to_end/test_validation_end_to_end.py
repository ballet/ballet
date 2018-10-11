import os
import tempfile
from unittest.mock import patch
from subprocess import check_call
from textwrap import dedent
from types import ModuleType

import git
import numpy as np
import pandas as pd
from sklearn_pandas import DataFrameMapper

from ballet.compat import pathlib, safepath
from ballet.eng.misc import IdentityTransformer
from ballet.feature import Feature
from ballet.quickstart import generate_project
from ballet.util import get_enum_values
from ballet.util.git import switch_to_new_branch
from ballet.util.log import logger
from ballet.util.mod import import_module_at_path, modname_to_relpath
from ballet.validation import TEST_TYPE_ENV_VAR, BalletTestTypes


def test_end_to_end():
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
        return_value=[Feature(input='A', transformer=IdentityTransformer())]
    ):
        X_df = pd.util.testing.makeCustomDataframe(5, 2)
        X_df.columns = ['A', 'B']
        X, mapper = foo_features_buildfeatures.build_features(X_df)
        assert np.shape(X) == (5, 1)
        assert isinstance(mapper, DataFrameMapper)

    # write a new version of foo.load_data.load_data
    new_load_data_str = dedent("""
        import pandas as pd
        import sklearn.datasets
        def load_data():
            data = sklearn.datasets.load_boston()
            X_df = pd.DataFrame(data=data.data, columns=data.feature_names)
            y_df = pd.Series(data.target, name='price')
            return X_df, y_df
    """).strip()

    p = base.joinpath(modname, 'load_data.py')
    with p.open('w') as f:
        f.write(new_load_data_str)

    # commit changes
    repo = git.Repo(safepath(base))
    repo.index.add([str(p)])
    repo.index.commit('Load boston dataset')

    # call different validation routines
    def call_validate(ballet_test_type):
        with patch.dict(os.environ,
                        {TEST_TYPE_ENV_VAR: ballet_test_type}):
            check_call('./validate.py', cwd=safepath(base), env=os.environ)

    def call_validate_all():
        for ballet_test_type in get_enum_values(BalletTestTypes):
            call_validate(ballet_test_type)

    call_validate_all()

    # write a new feature
    new_feature_str = dedent("""
        import numpy as np
        from ballet import Feature
        from ballet.eng.base import SimpleFunctionTransformer
        
        input = 'DIS'
        transformer = SimpleFunctionTransformer(np.log)
    """).strip()

    p = base.joinpath(modname, 'features', 'contrib', 'user_bob',
                      'feature_log_dis.py')
    p.parent.mkdir(exist_ok=True)
    with p.open('w') as f:
        f.write(new_feature_str)
    p1 = p.parent.joinpath('__init__.py')
    p1.touch()

    # commit new feature on master
    repo.index.add([str(p), str(p1)])
    repo.index.commit('Add log(DIS) feature')

    # call different validation routines
    call_validate_all()

    # branch to a fake PR
    switch_to_new_branch(repo, 'pull/1')

    # write a new feature
    new_feature_str1 = new_feature_str.replace('DIS', 'TAX')
    p2 = p.parent.joinpath('feature_log_tax.py')
    with p2.open('w') as f:
        f.write(new_feature_str1)
    repo.index.add([str(p2)])
    repo.index.commit('Add log(TAX) feature')

    # call different validation routines
    call_validate_all()

    _tempdir.cleanup()


if __name__ == '__main__':
    import logging
    import ballet.util.log
    ballet.util.log.enable(level=logging.INFO)

    test_end_to_end()
