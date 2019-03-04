import random
import subprocess
import tempfile

import funcy
import git
from funcy import any_fn, contextmanager, merge
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn_pandas.pipeline import TransformerPipeline

from ballet.compat import pathlib
from ballet.eng.misc import IdentityTransformer
from ballet.util.git import set_config_variables
from ballet.util.log import logger

EPSILON = 1e-4


class FragileTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, bad_input_checks, errors):
        """Raises a random error if any input check returns True"""
        super().__init__()

        self._check = any_fn(*bad_input_checks)
        self._errors = errors

        self._random = random.Random()
        self._random.seed = hash(merge(bad_input_checks, errors))

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


def make_mock_commit(repo, kind='A', path=None, content=None):
    '''Commits one file to repo'''
    if not path:
        path = 'file{}'.format(random.randint(0, 999))

    dir = repo.working_tree_dir
    abspath = pathlib.Path(dir).joinpath(path)

    if kind == 'A':
        # TODO make robust
        abspath.parent.mkdir(parents=True, exist_ok=True)

        if abspath.exists():
            # because this would be a kind=='M'
            raise FileExistsError(str(abspath))
        else:
            if content is not None:
                with abspath.open('w') as f:
                    f.write(content)
            else:
                abspath.touch()
        repo.git.add(str(abspath))
        repo.git.commit(m='Commit {}'.format(str(abspath)))
    else:
        raise NotImplementedError

    return repo.head.commit


def make_mock_commits(repo, n=10, filename='file{i}.py'):
    '''Create n sequential files/commits'''
    if '{i}' not in filename:
        raise ValueError

    commits = []
    for i in range(n):
        commit = make_mock_commit(repo, path=filename.format(i=i))
        commits.append(commit)
    return commits


def set_ci_git_config_variables(repo):
    set_config_variables(repo, {
        'user.name': 'Foo Bar',
        'user.email': 'foo@bar.com',
    })


@contextmanager
def mock_repo():
    '''Create a new repo'''
    with tempfile.TemporaryDirectory() as tmpdir:
        dir = pathlib.Path(tmpdir)
        repo = git.Repo.init(str(dir))
        set_ci_git_config_variables(repo)
        yield repo


def tree(dir):
    with funcy.suppress((FileNotFoundError, subprocess.SubprocessError)):
        cmd = ['tree', '-A', '-n', '--charset', 'ASCII', str(dir)]
        logger.debug('Popen({cmd!r})'.format(cmd=cmd))
        tree_output = subprocess.check_output(cmd).decode()
        logger.debug(tree_output)
        return tree_output
