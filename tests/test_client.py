from unittest.mock import patch, PropertyMock
from ballet.project import FeatureEngineeringProject
from ballet.eng import IdentityTransformer, NullFiller
from ballet.client import Client
from ballet.feature import Feature
import pytest


@pytest.fixture
def feature_engineering_project(sample_data):
    def load_data():
        return sample_data.X, sample_data.y

    encoder = IdentityTransformer()

    features = [
        Feature(
            'size', NullFiller(0),
            source='foo.features.contrib.user_x.feature_y'
        )
    ]

    with patch(
        'ballet.project.FeatureEngineeringProject.features',
        new_callable=PropertyMock
    ) as mock_features:
        fe_project = FeatureEngineeringProject(
            package=None,
            encoder=encoder,
            load_data=load_data,
        )
        mock_features.return_value = features
        yield fe_project


@pytest.fixture
def client(feature_engineering_project):
    with patch(
        'ballet.client.Client.api',
        new_callable=PropertyMock
    ) as mock_api:
        client = Client(None)
        mock_api.return_value = feature_engineering_project
        yield client


def test_discover_features(client):
    features = client.api.features

    df = client.discover()

    expected_cols = {
        'name', 'description', 'input', 'transformer', 'output', 'author',
        'source', 'mutual_information', 'conditional_mutual_information',
        'average_variance', 'average_nunique',
    }
    actual_cols = df.columns
    assert not expected_cols.symmetric_difference(actual_cols)

    assert df.shape[0] == len(features)
