from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ballet.eng.base import SimpleFunctionTransformer
from ballet.eng.misc import IdentityTransformer
from ballet.feature import Feature
from ballet.util import asarray2d
from ballet.validation.feature_pruning.validator import (
    GFSSFPruner, NoOpPruner, RandomPruner,)
from tests.util import load_regression_data


def test_noop_pruner():
    X_df = y_df = y = None
    existing_features = []
    feature = None

    expected = []

    pruner = NoOpPruner(X_df, y_df, y, existing_features, feature)
    actual = pruner.prune()

    assert expected == actual


@patch('random.choice')
@patch('random.uniform', return_value=0.0)  # makes sure pruning is performed
def test_random_pruner(mock_uniform, mock_choice):
    existing_features = [MagicMock(), MagicMock()]

    mock_choice.return_value = existing_features[0]  # the pruned feature
    expected = [existing_features[0]]

    X_df = y_df = y = None
    feature = None

    pruner = RandomPruner(X_df, y_df, y, existing_features, feature)
    actual = pruner.prune()

    assert expected == actual


@pytest.fixture
def sample_data():
    X_df, y_df = load_regression_data(
        n_informative=1, n_uninformative=14)
    y = y_df
    return X_df, y_df, y


def test_gfssf_pruner_prune_exact_replicas(sample_data):
    X_df, y_df, y = sample_data

    feature_1 = Feature(
        input='A_0',
        transformer=IdentityTransformer(),
        source='1st Feature')
    feature_2 = Feature(
        input='A_0',
        transformer=IdentityTransformer(),
        source='2nd Feature')
    gfssf_pruner = GFSSFPruner(
        X_df, y_df, y, [feature_1], feature_2)

    redunant_features = gfssf_pruner.prune()
    assert feature_1 in redunant_features, \
        'Exact replica features should be pruned'


@pytest.mark.xfail
def test_gfssf_pruner_prune_weak_replicas(sample_data):
    X_df, y_df, y = sample_data

    def add_noise(X):
        X = asarray2d(X)
        return X + np.random.normal(0, 0.5, X.shape)

    noise_transformer = SimpleFunctionTransformer(add_noise)
    feature_weak = Feature(
        input='A_0',
        transformer=noise_transformer,
        source='1st Feature')
    feature_strong = Feature(
        input='A_0',
        transformer=IdentityTransformer(),
        source='2nd Feature')
    gfssf_pruner = GFSSFPruner(
        X_df, y_df, y, [feature_weak], feature_strong)

    redunant_features = gfssf_pruner.prune()
    assert feature_weak in redunant_features, \
        'Noisy features should be pruned'


def test_gfssf_pruner_keep_relevant(sample_data):
    X_df, y_df, y = sample_data

    feature_1 = Feature(
        input='A_0',
        transformer=IdentityTransformer(),
        source='1st Feature')
    feature_2 = Feature(
        input='Z_0',
        transformer=IdentityTransformer(),
        source='2nd Feature')
    gfssf_pruner = GFSSFPruner(
        X_df, y_df, y, [feature_1], feature_2)

    redunant_features = gfssf_pruner.prune()
    assert feature_1 not in redunant_features, \
        'Still relevant features should be pruned'


@pytest.mark.xfail
def test_gfssf_pruner_prune_irrelevant_features(sample_data):
    X_df, y_df, y = sample_data

    feature_1 = Feature(
        input='Z_0',
        transformer=IdentityTransformer(),
        source='1st Feature')
    feature_2 = Feature(
        input='A_0',
        transformer=IdentityTransformer(),
        source='2nd Feature')
    gfssf_pruner = GFSSFPruner(
        X_df, y_df, y, [feature_1], feature_2)

    redunant_features = gfssf_pruner.prune()
    assert feature_1 in redunant_features, \
        'Irrelevant features should be pruned'
