import pandas as pd
import pytest

from ballet.encoder import make_robust_encoder
from ballet.eng import IdentityTransformer
from ballet.util.testing import assert_series_equal

with_encoder = pytest.mark.parametrize(
    'encoder',
    [
        None,
        lambda x: x,
        IdentityTransformer(),
        [None],
        [None, None],
    ]
)


@with_encoder
def test_encoder_pipeline(encoder):
    encoder_pipeline = make_robust_encoder(encoder)
    y_df = pd.util.testing.makeFloatSeries()
    encoder_pipeline.fit(y_df)
    y = encoder_pipeline.transform(y_df)
    assert_series_equal(y, y_df)
