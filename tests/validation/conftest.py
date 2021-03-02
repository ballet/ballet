import pathlib
from typing import NamedTuple
from unittest.mock import MagicMock, create_autospec, patch

import numpy as np
import pandas as pd
import pytest

from ballet.project import Project
from ballet.validation.common import ChangeCollector


class SampleData(NamedTuple):
    df: pd.DataFrame
    X: pd.DataFrame
    y: pd.DataFrame


@pytest.fixture
def sample_data():
    df = pd.DataFrame(
        data={
            'country': ['USA', 'USA', 'Canada', 'Japan'],
            'year': [2001, 2002, 2001, 2002],
            'size': [np.nan, -11, 12, 0.0],
            'strength': [18, 110, np.nan, 101],
            'happy': [False, True, False, False]
        }
    ).set_index(['country', 'year'])
    X = df[['size', 'strength']]
    y = df[['happy']]
    return SampleData(df, X, y)


def make_mock_project(repo, pr_num, path, contrib_module_path):
    def mock_get(key):
        if key == 'contrib.module_path':
            return contrib_module_path
        else:
            raise KeyError
    config = MagicMock()
    config.get.side_effect = mock_get

    project = create_autospec(Project)
    project.repo = repo
    project.pr_num = str(pr_num)
    project.path = pathlib.Path(path)
    project.config = config

    # in attr map
    project.api = MagicMock()

    return project


@pytest.fixture
def null_change_collector(mock_repo):
    repo = mock_repo
    pr_num = 3

    commit_range = 'HEAD^..HEAD'
    contrib_module_path = None

    travis_env_vars = {
        'TRAVIS_BUILD_DIR': repo.working_tree_dir,
        'TRAVIS_PULL_REQUEST': str(pr_num),
        'TRAVIS_COMMIT_RANGE': commit_range,
    }

    with patch.dict('os.environ', travis_env_vars, clear=True):
        project_path = repo.working_tree_dir
        project = make_mock_project(repo, pr_num, project_path,
                                    contrib_module_path)
        yield ChangeCollector(project)
