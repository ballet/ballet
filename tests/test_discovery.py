import numpy as np
import pytest

import ballet.discovery
from ballet.discovery import countunique, discover
from ballet.eng import NullFiller
from ballet.feature import Feature
from ballet.util.testing import assert_array_equal
from tests.util import FragileTransformer


@pytest.mark.parametrize(
    'z,expected',
    [
        (np.array([[1, 2, 3]]).T, np.array([3])),
        (np.array([[1, 2, 3], [2, 2, 3]]).T, np.array([3, 2])),
    ]
)
def test_countunique(z, expected):
    nunique = countunique(z)
    assert_array_equal(nunique, expected)


def test_discover(sample_data):
    features = [
        Feature(
            'size', NullFiller(0),
            source='foo.features.contrib.user_a.feature_1'
        ),
        Feature(
            'strength', NullFiller(100),
            source='foo.features.contrib.user_b.feature_1'
        )
    ]
    X_df, y_df = sample_data.X, sample_data.y
    y = np.asfarray(y_df)

    df = discover(features, X_df, y_df, y)

    expected_cols = {
        'name', 'description', 'input', 'transformer', 'primitives', 'output',
        'author', 'source', 'mutual_information',
        'conditional_mutual_information', 'mean', 'std', 'variance', 'nunique',
    }
    actual_cols = df.columns
    assert not expected_cols.symmetric_difference(actual_cols)

    assert df.shape[0] == len(features)

    # test filter
    input = 'size'
    df = discover(features, X_df, y_df, y, input=input)
    assert df.shape[0] == len([
        feature
        for feature in features
        if feature.input == input or input in feature.input
    ])

    # test no data available
    # have to clear cache, as values on data already known
    ballet.discovery._summarize_feature.memory.clear()
    df = discover(features, None, None, None)
    assert df.shape[0] == len(features)
    actual_cols = df.columns
    assert not expected_cols.symmetric_difference(actual_cols)
    assert np.isnan(df['mean'].at[0])


def test_discover_feature_error(sample_data):
    features = [
        Feature('size', FragileTransformer()),
    ]
    X_df, y_df = sample_data.X, sample_data.y
    y = np.asfarray(y_df)

    df = discover(features, X_df, y_df, y)

    assert df.shape[0] == len(features)
    assert np.isnan(df['mean'].at[0])
