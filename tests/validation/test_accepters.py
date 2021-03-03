from unittest.mock import patch

import pytest

from ballet.eng.misc import IdentityTransformer
from ballet.feature import Feature
from ballet.validation.feature_acceptance.validator import (
    GFSSFAccepter, NeverAccepter, RandomAccepter,)
from tests.util import load_regression_data


def test_noop_accepter():
    X_df = y_df = y = None
    existing_features = []
    feature = None

    expected = False

    accepter = NeverAccepter(X_df, y_df, y, existing_features, feature)
    actual = accepter.judge()

    assert expected == actual


@patch('random.uniform', return_value=0.0)  # makes sure feature is accepted
def test_random_accepter(mock_uniform):
    X_df = y_df = y = None
    existing_features = []
    candidate_feature = None

    expected = True

    accepter = RandomAccepter(
        X_df, y_df, y, existing_features, candidate_feature)
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
        X_df, y_df, y, features, candidate_feature)

    assert accepter is not None
