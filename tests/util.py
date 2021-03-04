import pathlib
import random
import subprocess

from funcy import any_fn, ignore, merge
from sklearn_pandas.pipeline import TransformerPipeline

from ballet.eng.base import BaseTransformer
from ballet.eng.misc import IdentityTransformer
from ballet.util.git import set_config_variables
from ballet.util.log import logger

EPSILON = 1e-4


class FragileTransformer(BaseTransformer):

    def __init__(self, bad_input_checks=None, errors=None):
        """Raises a random error if any input check returns True"""
        self.bad_input_checks = bad_input_checks
        self.errors = errors

        self._check = any_fn(*self.bad_input_checks)
        seed = hash(merge(self.bad_input_checks, self.errors))
        self._random = random.Random(seed)

    def _raise(self):
        raise self._random.choice(self.errors)

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
            (f'IdentityTransformer{i:02d}', IdentityTransformer())
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


def make_mock_commit(repo, kind='A', path=None, content=None):
    """Commits one file to repo"""
    if not path:
        path = 'file{}'.format(random.randint(0, 999))

    dir = repo.working_tree_dir
    abspath = pathlib.Path(dir).joinpath(path).resolve()

    if kind == 'A':
        # TODO make robust
        abspath.parent.mkdir(parents=True, exist_ok=True)

        if abspath.exists():
            # because this would be a kind=='M'
            raise FileExistsError(str(abspath))
        else:
            with abspath.open('w') as f:
                f.write(content or '')
        repo.git.add(str(abspath))
        repo.git.commit(m=f'Commit {abspath}')
    else:
        raise NotImplementedError

    return repo.head.commit


def make_mock_commits(repo, n=10, filename='file{i}.py'):
    """Create n sequential files/commits"""
    if '{i}' not in filename:
        raise ValueError

    commits = []
    for i in range(n):
        commit = make_mock_commit(repo, path=filename.format(i=i))
        commits.append(commit)
    return commits


def set_ci_git_config_variables(repo, name='Foo Bar', email='foo@bar.com'):
    set_config_variables(repo, {
        'user.name': name,
        'user.email': email,
    })


@ignore((FileNotFoundError, subprocess.CalledProcessError))
def tree(dir):
    cmd = ['tree', '-A', '-n', '--charset', 'ASCII', str(dir)]
    logger.debug(f'Popen({cmd!r})')
    tree_output = subprocess.check_output(cmd).decode()
    logger.debug(tree_output)
    return tree_output


def load_regression_data(n_informative=1, n_uninformative=14, n_samples=500):
    import pandas as pd
    from sklearn.datasets import make_regression

    q = n_informative
    p = n_informative + n_uninformative
    X, y, coef = make_regression(
        n_samples=n_samples, n_features=p, n_informative=q, coef=True,
        shuffle=True, random_state=1)

    # informative columns are 'A_0', 'A_1', ...
    informative = [f'A_{i}' for i in range(n_informative)]

    # uninformative columns are 'Z_0', 'Z_1', ...
    uninformative = [f'Z_{i}' for i in range(n_uninformative)]

    columns = [
        informative.pop(0) if c != 0 else uninformative.pop(0)
        for c in coef
    ]

    X_df = pd.DataFrame(data=X, columns=columns)
    y_df = pd.Series(y)
    return X_df, y_df
