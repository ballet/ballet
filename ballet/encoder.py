from sklearn_pandas.pipeline import TransformerPipeline

import ballet.transformer
from ballet.util.typing import OneOrMore, TransformerLike


class EncoderPipeline(TransformerPipeline):
    """Pipeline of target encoder steps

    This wraps sklearn.pipeline.Pipeline. Each step receives a single argument
    ``y`` to their fit and transform methods. This is needed because some
    consumers like MLBlocks passes arguments by keyword, and we need to pass an
    argument named ``y`` rather than one named ``X``.

    Args:
        can_skip_transform_none: behavior if during the transform stage, the
            input y is None (as would be the case during the ``predict`` stage
            of an MLPipeline). If false (the default), then we call the
            pipeline's transform method on y. If true, we skip calling the
            transform method and instead return immediately (returning the
            value ``None``).
    """

    def __init__(self, *args, can_skip_transform_none=False, **kwargs):
        self.can_skip_transform_none = can_skip_transform_none
        super().__init__(*args, **kwargs)

    def fit(self, y, **fit_params):
        return super().fit(X=y, **fit_params)

    def transform(self, y):
        if self.can_skip_transform_none and y is None:
            return None
        return super().transform(X=y)

    def fit_transform(self, y, **fit_params):
        return self.fit(y, **fit_params).transform(y)


def make_encoder_pipeline(steps, **kwargs):
    return EncoderPipeline(
        ballet.transformer._name_estimators(steps), **kwargs)


def make_robust_encoder(
    steps: OneOrMore[TransformerLike],
    **kwargs,
):
    if not isinstance(steps, list):
        steps = [steps]
    steps = [
        ballet.transformer.make_robust_transformer(step)
        for step in steps
    ]
    return make_encoder_pipeline(steps, **kwargs)
