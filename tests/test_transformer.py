import funcy
import numpy as np
import pandas as pd
import pytest
import sklearn.preprocessing

from ballet.compat import SimpleImputer
from ballet.eng.misc import IdentityTransformer
from ballet.transformer import (
    DelegatingRobustTransformer, make_robust_transformer,)
from ballet.util import asarray2d

from .util import FragileTransformer, FragileTransformerPipeline


@pytest.fixture
def sample_data():
    X_ser = pd.util.testing.makeFloatSeries()
    X_df = X_ser.to_frame()
    X_arr1d = np.asarray(X_ser)
    X_arr2d = np.asarray(X_df)
    y_ser = X_ser.copy()
    y_df = X_df.copy()
    y_arr1d = np.asarray(y_ser)
    y_arr2d = np.asarray(y_df)
    d = {
        'ser': (X_ser, y_ser),
        'df': (X_df, y_df),
        'arr1d': (X_arr1d, y_arr1d),
        'arr2d': (X_arr2d, y_arr2d),
    }
    return d


@pytest.mark.parametrize(
    'input_types, bad_input_checks, catches',
    [
        (
            ('ser', ),
            (lambda x: isinstance(x, pd.Series), ),
            (ValueError, TypeError),
        ),
        (
            ('ser', 'df', ),
            (
                lambda x: isinstance(x, pd.Series),
                lambda x: isinstance(x, pd.DataFrame)
            ),
            (ValueError, TypeError),
        ),
        (
            ('ser', 'df', 'arr1d'),
            (
                lambda x: isinstance(x, pd.Series),
                lambda x: isinstance(x, pd.DataFrame),
                lambda x: isinstance(x, np.ndarray) and x.ndim == 1,
            ),
            (ValueError, TypeError),
        )
    ],
    ids=[
        'ser',
        'df',
        'arr',
    ]
)
@pytest.mark.parametrize(
    'transformer_maker',
    [
        FragileTransformer,
        funcy.partial(FragileTransformerPipeline, 3)
    ]
)
def test_robust_transformer(
    input_types, bad_input_checks, catches, transformer_maker,
    sample_data,
):
    fragile_transformer = transformer_maker(bad_input_checks, catches)
    robust_transformer = DelegatingRobustTransformer(
        transformer_maker(bad_input_checks, catches))

    for input_type in input_types:
        X, y = sample_data[input_type]
        # fragile transformer raises error
        with pytest.raises(catches):
            fragile_transformer.fit_transform(X, y)
        # robust transformer does not raise error
        X_robust = robust_transformer.fit_transform(X, y)
        assert np.array_equal(
            asarray2d(X),
            asarray2d(X_robust)
        )


def test_robust_transformer_sklearn(sample_data):
    Transformers = (
        SimpleImputer,
        sklearn.preprocessing.StandardScaler,
        sklearn.preprocessing.Binarizer,
        sklearn.preprocessing.PolynomialFeatures,
    )
    # some of these input types are bad for sklearn.
    input_types = ('ser', 'df', 'arr1d')
    for Transformer in Transformers:
        robust_transformer = DelegatingRobustTransformer(Transformer())
        for input_type in input_types:
            X, y = sample_data[input_type]
            robust_transformer.fit_transform(X, y=y)


@pytest.mark.parametrize(
    'robust_maker',
    [
        DelegatingRobustTransformer,
        lambda x: make_robust_transformer([x]),
    ]
)
def test_robust_str_repr(robust_maker):
    robust_transformer = robust_maker(IdentityTransformer())
    for func in [str, repr]:
        s = func(robust_transformer)
        assert len(s) > 0
