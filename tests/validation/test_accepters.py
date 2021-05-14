from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from ballet.eng.misc import IdentityTransformer
from ballet.feature import Feature
from ballet.validation.feature_acceptance.validator import (
    CompoundAccepter, GFSSFAccepter, MutualInformationAccepter, NeverAccepter,
    RandomAccepter, VarianceThresholdAccepter,)
from tests.util import load_regression_data


def test_noop_accepter():
    X_df = y_df = y = None
    existing_features = []
    feature = None

    expected = False

    accepter = NeverAccepter(X_df, y_df, X_df, y, existing_features, feature)
    actual = accepter.judge()

    assert expected == actual


@patch('random.uniform', return_value=0.0)  # makes sure feature is accepted
def test_random_accepter(mock_uniform):
    X_df = y_df = y = None
    existing_features = []
    candidate_feature = None

    expected = True

    accepter = RandomAccepter(
        X_df, y_df, X_df, y, existing_features, candidate_feature)
    actual = accepter.judge()

    assert expected == actual


@pytest.fixture
def sample_data():
    X_df, y_df = load_regression_data(
        n_informative=1, n_uninformative=14)
    y = y_df
    return X_df, y_df, y


def test_gfssf_accepter_init(sample_data):
    X_df, y_df, y = sample_data

    feature_1 = Feature(
        input='A_0',
        transformer=IdentityTransformer(),
        source='1st Feature')
    feature_2 = Feature(
        input='Z_0',
        transformer=IdentityTransformer(),
        source='2nd Feature')

    features = [feature_1]
    candidate_feature = feature_2

    accepter = GFSSFAccepter(
        X_df, y_df, X_df, y, features, candidate_feature)

    assert accepter is not None


@patch('numpy.var', return_value=0.0)
def test_variance_threshold_accepter(mock_var, sample_data):
    expected = False
    X_df, y_df, y = sample_data
    feature = Feature(
        input='A_0',
        transformer=IdentityTransformer(),
        source='1st Feature')
    accepter = VarianceThresholdAccepter(
        X_df, y_df, X_df, y, [], feature)
    actual = accepter.judge()

    assert expected == actual


def test_variance_threshold_accepter_feature_group():
    expected = True
    # variance is 0.25 per column, > 0.05 threshold
    X = pd.DataFrame(np.eye(2))
    y = None
    feature = Feature(
        input=[0, 1],
        transformer=IdentityTransformer(),
        source='1st Feature')
    accepter = VarianceThresholdAccepter(
        X, y, X, y, [], feature)
    actual = accepter.judge()

    assert expected == actual


@patch(
    'ballet.validation.feature_acceptance.validator.estimate_mutual_information',  # noqa
    return_value=0.99
)
def test_mutual_information_accepter(_, sample_data):
    expected = True
    X_df, y_df, y = sample_data
    feature = Feature(
        input='A_0',
        transformer=IdentityTransformer(),
        source='1st Feature')
    accepter = MutualInformationAccepter(
        X_df, y_df, X_df, y, [], feature)
    actual = accepter.judge()

    assert expected == actual


@pytest.mark.parametrize(
    'handle_nan_targets, expected',
    [
        ('fail', False),
        ('ignore', True)
    ],
)
def test_mutual_information_accepter_nans(handle_nan_targets, expected):
    X_df = pd.DataFrame({'A': [1, 2, 3]})
    y = np.array([np.nan, 2, 3]).reshape(-1, 1)
    feature = Feature(
        input='A',
        transformer=IdentityTransformer())
    accepter = MutualInformationAccepter(
        X_df, y, X_df, y, [], feature, handle_nan_targets=handle_nan_targets)
    actual = accepter.judge()
    assert expected == actual


def test_compound_accepter(sample_data):
    expected = False
    X_df, y_df, y = sample_data
    agg = 'all'
    specs = [
        'ballet.validation.feature_acceptance.validator.AlwaysAccepter',
        {
            'name': 'ballet.validation.feature_acceptance.validator.RandomAccepter',  # noqa
            'params': {
                'p': 0.00,
            }
        }
    ]
    feature = Feature(
        input='A_0',
        transformer=IdentityTransformer(),
        source='1st Feature')
    accepter = CompoundAccepter(
        X_df, y_df, X_df, y, [], feature, agg=agg, specs=specs
    )
    actual = accepter.judge()

    assert expected == actual
