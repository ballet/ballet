import funcy
import pytest

from ballet.eng.misc import IdentityTransformer
from ballet.feature import Feature
from ballet.pipeline import FeatureEngineeringPipeline


@pytest.fixture(
    params=[
        'input',
        ['input'],
        lambda df: 'input',
        lambda df: ['input'],
    ],
    ids=[
        'string',
        'list of string',
        'callable to string',
        'callable to list of string',
    ]
)
def inputs(request):
    input = request.param
    transformer = IdentityTransformer()
    return input, transformer


def test_feature_init_with_one_transformer(inputs):
    input, transformer = inputs
    Feature(input, transformer)


def test_feature_init_with_list_of_transformer(inputs):
    input, transformer = inputs
    Feature(input, [transformer, transformer])


def test_feature_init_with_callable(inputs):
    input, transformer = inputs
    Feature(input, funcy.identity)


def test_feature_init_with_list_of_transformers_and_callables(inputs):
    input, transformer = inputs
    Feature(input, [transformer, funcy.identity])


def test_feature_init_with_none(inputs):
    input, transformer = inputs
    Feature(input, None)


def test_feature_init_with_list_of_none(inputs):
    input, transformer = inputs
    Feature(input, [None, None])


def test_feature_init_with_list_of_none_and_notnone(inputs):
    input, transformer = inputs
    Feature(input, [None, transformer])


def test_feature_init_invalid_transformer_api(inputs):
    input, transformer = inputs
    with pytest.raises(ValueError):
        Feature(input, object())

    with pytest.raises(ValueError):
        Feature(input, IdentityTransformer)


def test_feature_repr(inputs):
    input, transformer = inputs
    feature = Feature(input, transformer)
    assert isinstance(repr(feature), str)


def test_feature_as_input_transformer_tuple(inputs):
    input, transformer = inputs
    feature = Feature(input, transformer)
    tup = feature.as_input_transformer_tuple()
    assert isinstance(tup, tuple)
    assert len(tup) == 3


def test_feature_as_feature_engineering_pipeline(inputs):
    input, transformer = inputs
    feature = Feature(input, transformer)
    mapper = feature.as_feature_engineering_pipeline()
    assert isinstance(mapper, FeatureEngineeringPipeline)


def test_feature_pipeline(inputs):
    input, transformer = inputs
    feature = Feature(input, transformer)
    pipeline = feature.pipeline
    assert isinstance(pipeline, FeatureEngineeringPipeline)
    pipeline2 = feature.pipeline
    assert pipeline is pipeline2


@pytest.mark.parametrize(
    'name,output,expected_output',
    [
        (None, 'baz', 'baz'),
        (None, ['foobaz', 'barbaz'], ['foobaz', 'barbaz']),
        (None, None, None),
        ('baz', None, 'baz'),
        ('Some long name', None, 'some_long_name'),
    ]
)
def test_feature_output(inputs, name, output, expected_output):
    input, transformer = inputs
    feature = Feature(input, transformer, name=name, output=output)
    assert feature.output == expected_output
