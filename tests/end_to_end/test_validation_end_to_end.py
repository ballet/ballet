import os
from subprocess import check_call
from textwrap import dedent
from types import ModuleType
from unittest.mock import patch

import git
import numpy as np
import pandas as pd
from sklearn_pandas import DataFrameMapper

from ballet.compat import safepath
from ballet.eng.misc import IdentityTransformer
from ballet.feature import Feature
from ballet.quickstart import generate_project
from ballet.util import get_enum_values
from ballet.util.git import switch_to_new_branch
from ballet.util.log import logger
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


def make_feature_str(input):
    return dedent("""
        from ballet import Feature
        from ballet.eng.misc import IdentityTransformer
        input = '{input}'
        transformer = IdentityTransformer()
        feature = Feature(input, transformer)
    """.format(input=input)).strip()


def test_end_to_end(tempdir):
    modname = 'foo'
    extra_context = {
        'project_name': modname.capitalize(),
        'project_slug': modname,
    }

    generate_project(no_input=True, extra_context=extra_context,
                     output_dir=tempdir)

    # make sure we can import different modules without error
    base = tempdir.joinpath(modname)

    def _import(modname):
        relpath = modname_to_relpath(modname)
        abspath = base.joinpath(relpath)
        return import_module_at_path(modname, abspath)

    foo = _import('foo')
    assert isinstance(foo, ModuleType)

    foo_features = _import('foo.features')
    assert isinstance(foo_features, ModuleType)

    get_contrib_features = foo_features.get_contrib_features
    features = get_contrib_features()
    assert len(features) == 0

    # first providing a mock feature, call build
    with patch.object(
        foo_features, 'get_contrib_features',
        return_value=[Feature(input='A', transformer=IdentityTransformer())]
    ):
        X_df = pd.util.testing.makeCustomDataframe(5, 2)
        X_df.columns = ['A', 'B']
        out = foo_features.build(X_df=X_df, y_df=[])
        assert np.shape(out['X']) == (5, 1)
        assert isinstance(out['mapper_X'], DataFrameMapper)

    # write a new version of foo.load_data.load_data
    new_load_data_str = dedent("""
        import pandas as pd
        from sklearn.datasets import make_regression

        def load_data():
            p = 15
            q = 2
            X, y, coef = make_regression(
                n_samples=50, n_features=p, n_informative=q, coef=True,
                shuffle=True, random_state=1)

            # informative columns are 'A', 'B'
            # uninformative columns are 'Z_0', ..., 'Z_11'
            columns = []
            informative = list('AB')
            other = ['Z_{i}'.format(i=i) for i in reversed(range(p-q))]
            for i in range(p):
                if coef[i] == 0:
                    columns.append(other.pop())
                else:
                    columns.append(informative.pop())

            X_df = pd.DataFrame(data=X, columns=columns)
            y_df = pd.Series(y)
            return X_df, y_df
    """).strip()

    p = base.joinpath(modname, 'load_data.py')
    with p.open('w') as f:
        f.write(new_load_data_str)

    # commit changes
    repo = git.Repo(safepath(base))
    repo.index.add([str(p)])
    repo.index.commit('Load mock regression dataset')

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

    # new_feature_str = make_feature_str('Z_0')
    # username = 'alice'
    # featurename = 'Z_0'
    # submit_feature(repo, contrib_dir, username, featurename, new_feature_str)

    # # call different validation routines
    call_validate_all()

    # branch to a fake PR and write a new feature
    logger.info('Switching to pull request 1, User Bob, Feature A')
    switch_to_new_branch(repo, 'pull/1')
    new_feature_str = make_feature_str('A')
    username = 'bob'
    featurename = 'A'
    submit_feature(repo, contrib_dir, username, featurename, new_feature_str)

    # call different validation routines
    logger.info('Validating pull request 1, User Bob, Feature A')
    call_validate_all(pr=1)

    # merge PR with master
    logger.info('Merging into master')
    repo.git.checkout('master')
    repo.git.merge('pull/1', no_ff=True)

    # call different validation routines
    logger.info('Validating after merge')
    call_validate_all()

    # write another new feature
    logger.info('Switching to pull request 2, User Charlie, Feature Z_1')
    switch_to_new_branch(repo, 'pull/2')
    new_feature_str = make_feature_str('Z_1')
    username = 'charlie'
    featurename = 'Z_1'
    submit_feature(repo, contrib_dir, username, featurename, new_feature_str)

    # if we expect this feature to succeed -- with NoOpAcceptanceEvaluator
    call_validate_all(pr=2)

    # if we expect this feature to fail -- with a more reasonable evaluator
    # with pytest.raises(CalledProcessError):
    #     logger.info('Validating pull request 2, User Charlie, Feature Z_1')
    #     call_validate_all(pr=2)

    # write another new feature
    repo.git.checkout('master')
    switch_to_new_branch(repo, 'pull/3')
    new_feature_str = make_feature_str('B')
    username = 'charlie'
    featurename = 'B'
    submit_feature(repo, contrib_dir, username, featurename, new_feature_str)
    call_validate_all(pr=3)

    # merge PR with master
    repo.git.checkout('master')
    repo.git.merge('pull/3', no_ff=True)

    # call different validation routines
    call_validate_all()


if __name__ == '__main__':
    import ballet.util.log
    ballet.util.log.enable(level='INFO')

    test_end_to_end()
