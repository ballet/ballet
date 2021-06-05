from sklearn_pandas.pipeline import TransformerPipeline

import ballet.transformer
from ballet.util.typing import OneOrMore, TransformerLike


class EncoderPipeline(TransformerPipeline):
    """Pipeline of target encoder steps

    This wraps sklearn.pipeline.Pipeline. Each step receives a single argument
    ``y`` to their fit and transform methods. This is needed because some
    consumers like MLBlocks passes arguments by keyword, and we need to pass an
    argument named ``y`` rather than one named ``X``.
    """
    def fit(self, y, **fit_params):
        return super().fit(X=y, **fit_params)

    def transform(self, y):
        return super().transform(X=y)


def make_encoder_pipeline(steps):
    return EncoderPipeline(ballet.transformer._name_estimators(steps))


def make_robust_encoder(
    steps: OneOrMore[TransformerLike],
):
    if not isinstance(steps, list):
        steps = [steps]
    steps = [
        ballet.transformer.make_robust_transformer(step)
        for step in steps
    ]
    return make_encoder_pipeline(steps)
