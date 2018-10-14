import os
import tempfile
from subprocess import check_call
from textwrap import dedent
from types import ModuleType
from unittest.mock import patch

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
from ballet.util.mod import import_module_at_path, modname_to_relpath
from ballet.validation import TEST_TYPE_ENV_VAR, BalletTestTypes


def submit_feature(repo, contrib_dir, username, featurename, new_feature_str):
    feature_path = contrib_dir.joinpath(
        'user_{}'.format(username), 'feature_{}.py'.format(featurename))
    feature_path.parent.mkdir(exist_ok=True)
    init_path = feature_path.parent.joinpath('__init__.py')

    init_path.touch()
    with feature_path.open('w') as f:
        f.write(new_feature_str)

    repo.index.add([str(init_path), str(feature_path)])
    repo.index.commit('Add {} feature'.format(feature_path))


def test_end_to_end():
    modname = 'foo'
    extra_context = {
        'project_name': modname.capitalize(),
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

    def call_validate_all(pr=None):
        envvars = {
            'TRAVIS_BUILD_DIR': repo.working_tree_dir,
        }
        if pr is None:
            envvars['TRAVIS_PULL_REQUEST'] = 'false'
            envvars['TRAVIS_COMMIT_RANGE'] = ''
        else:
            envvars['TRAVIS_PULL_REQUEST'] = str(pr)
            envvars['TRAVIS_COMMIT_RANGE'] = '{master}..{commit}'.format(
                master='master',
                commit=repo.commit('pull/{}'.format(pr)).hexsha)

        with patch.dict(os.environ, envvars):
            for ballet_test_type in get_enum_values(BalletTestTypes):
                call_validate(ballet_test_type)

    call_validate_all()

    # write a new feature
    contrib_dir = base.joinpath(modname, 'features', 'contrib')

    new_feature_str = dedent("""
        import numpy as np
        from ballet import Feature
        from ballet.eng.base import SimpleFunctionTransformer
        input = 'DIS'
        transformer = SimpleFunctionTransformer(np.log)
        feature = Feature(input, transformer)
    """).strip()
    username = 'alice'
    featurename = 'log_dis'
    submit_feature(repo, contrib_dir, username, featurename, new_feature_str)

    # call different validation routines
    call_validate_all()

    # branch to a fake PR
    switch_to_new_branch(repo, 'pull/1')

    # write a new feature
    new_feature_str = new_feature_str.replace('DIS', 'TAX')
    username = 'bob'
    featurename = 'log_tax'
    submit_feature(repo, contrib_dir, username, featurename, new_feature_str)

    # call different validation routines
    call_validate_all(pr=1)

    # merge PR with master
    repo.git.checkout('master')
    repo.git.merge('pull/1', no_ff=True)

    # call different validation routines
    call_validate_all()

    _tempdir.cleanup()


if __name__ == '__main__':
    import logging
    import ballet.util.log
    ballet.util.log.enable(level=logging.INFO)

    test_end_to_end()
