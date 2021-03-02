import numpy as np
import pandas as pd
import pytest

from ballet.eng.misc import IdentityTransformer
from ballet.feature import Feature
from ballet.pipeline import FeatureEngineeringPipeline


@pytest.fixture
def inputs():
    input = 'foo'
    transformer = IdentityTransformer()
    return input, transformer


def test_init_seqcont(inputs):
    input, transformer = inputs
    feature = Feature(input, transformer)
    features = [feature]
    mapper = FeatureEngineeringPipeline(features)
    assert isinstance(mapper, FeatureEngineeringPipeline)


def test_init_scalar(inputs):
    input, transformer = inputs
    feature = Feature(input, transformer)
    mapper = FeatureEngineeringPipeline(feature)
    assert isinstance(mapper, FeatureEngineeringPipeline)


def test_fit(inputs):
    input, transformer = inputs
    feature = Feature(input, transformer)
    mapper = FeatureEngineeringPipeline(feature)
    df = pd.util.testing.makeCustomDataframe(5, 2)
    df.columns = ['foo', 'bar']
    mapper.fit(df)


def test_transform(inputs):
    input, transformer = inputs
    feature = Feature(input, transformer)
    mapper = FeatureEngineeringPipeline(feature)
    df = pd.util.testing.makeCustomDataframe(5, 2)
    df.columns = ['foo', 'bar']
    mapper.fit(df)
    X = mapper.transform(df)
    assert np.shape(X) == (5, 1)
