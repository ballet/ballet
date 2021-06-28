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


@pytest.mark.parametrize('expensive_stats', [True, False])
def test_discover(sample_data, expensive_stats):
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

    df = discover(features, X_df, y_df, y, expensive_stats=expensive_stats)

    expected_cols = {
        'name', 'description', 'input', 'transformer', 'primitives', 'output',
        'author', 'source', 'mutual_information',
        'conditional_mutual_information', 'ninputs', 'nvalues', 'ncontinuous',
        'ndiscrete', 'mean', 'std', 'variance', 'min', 'median', 'max',
        'nunique',
    }
    actual_cols = df.columns
    assert not expected_cols.symmetric_difference(actual_cols)

    assert df.shape[0] == len(features)

    # test filter
    input = 'size'
    discovery_df = discover(features, X_df, y_df, y, input=input)
    assert discovery_df.shape[0] == len([
        feature
        for feature in features
        if feature.input == input or input in feature.input
    ])

    # test no data available
    # have to clear cache, as values on data already known
    ballet.discovery._summarize_feature.memory.clear()
    discovery_df = discover(features, None, None, None)
    assert discovery_df.shape[0] == len(features)
    actual_cols = discovery_df.columns
    assert not expected_cols.symmetric_difference(actual_cols)
    assert np.isnan(discovery_df['mean'].at[0])


def test_discover_feature_error(sample_data):
    features = [
        Feature('size', FragileTransformer()),
    ]
    X_df, y_df = sample_data.X, sample_data.y
    y = np.asfarray(y_df)

    discovery_df = discover(features, X_df, y_df, y)

    assert discovery_df.shape[0] == len(features)
    assert np.isnan(discovery_df['mean'].at[0])


def test_discover_target_nans(sample_data):
    features = [
        Feature('size', NullFiller(0)),
    ]
    X_df, y_df = sample_data.X, sample_data.y
    y = np.asfarray(y_df)

    # introduce nan to target
    y[0] = np.nan

    discovery_df = discover(features, X_df, y_df, y)

    # stats with target should still be computed
    assert not np.isnan(discovery_df['mutual_information']).any()
