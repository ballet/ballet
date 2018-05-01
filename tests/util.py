import pathlib
import random
import tempfile

import funcy
import git
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn_pandas.pipeline import TransformerPipeline

from fhub_core.util import IdentityTransformer

EPSILON = 1e-4


class FragileTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, bad_input_checks, errors):
        '''Raises a random error if any input check returns True'''
        super().__init__()

        self._check = funcy.any_fn(*bad_input_checks)
        self._errors = errors

        self._random = random.Random()
        self._random.seed = hash(funcy.merge(bad_input_checks, errors))

    def _raise(self):
        raise self._random.choice(self._errors)

    def fit(self, X, y=None, **fit_kwargs):
        if self._check(X) or self._check(y):
            self._raise()

        return self

    def transform(self, X, **transform_kwargs):
        if self._check(X):
            self._raise()

        return X


class FragileTransformerPipeline(TransformerPipeline):
    def __init__(self, nsteps, bad_input_checks, errors, shuffle=True, seed=1):
        steps = [
            ('IdentityTransformer{:02d}'.format(i), IdentityTransformer())
            for i in range(nsteps - 1)
        ]
        fragile_transformer = FragileTransformer(bad_input_checks, errors)
        steps.append(
            (repr(fragile_transformer), fragile_transformer)
        )
        if shuffle:
            rand = random.Random()
            rand.seed(seed)
            rand.shuffle(steps)

        super().__init__(steps)


@funcy.contextmanager
def mock_commits(repo, n=10):
    '''Create n sequential files/commits'''
    dir = pathlib.Path(repo.working_tree_dir)
    commits = []
    for i in range(n):
        file = dir.joinpath('file{i}.py'.format(i=i))
        file.touch()
        repo.git.add(str(file))
        repo.git.commit(m='Commit {i}'.format(i=i))
        commits.append(repo.head.commit)
    yield commits


@funcy.contextmanager
def mock_repo():
    '''Create a new repo'''
    with tempfile.TemporaryDirectory() as tmpdir:
        dir = pathlib.Path(tmpdir)
        repo = git.Repo.init(dir)
        yield repo
