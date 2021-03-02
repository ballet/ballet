from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest


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
