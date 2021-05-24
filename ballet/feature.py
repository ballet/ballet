from inspect import signature
from typing import Optional, Tuple

from funcy import cached_property

import ballet.pipeline
from ballet.transformer import RobustTransformer, make_robust_transformer
from ballet.util.typing import OneOrMore, TransformerLike

__all__ = ('Feature', )


class Feature:
    """A feature definition

    Conceptually, a feature definition is a learned function that maps raw
    variables in one data instance to a vector of feature values. A feature
    definition can produce either a scalar feature value for each instance
    or a vector of feature values, as in the case of an embedding technique
    like PCA or the one-hot encoding of a categorical variable.

    Args:
        input: required columns from the input dataframe needed for the
            transformation
        transformer: transformer, sequence of transformers, or ``None``. A
            "transformer" is an instance of a class that provides a
            fit/transform-style learned transformation. Alternately, a
            callable can be provided, either by itself or in a list, in
            which case it will be converted into a
            :py:class:``FunctionTransformer`` for convenience. If ``None``
            is provided, it will be replaced with the
            :py:class:``IdentityTransformer``.
        name: name of the feature
        description: description of the feature
        output: ordered sequence of names of features
            produced by this transformer
        source: the module in which this feature was defined
        options: options
    """

    def __init__(
        self,
        input: OneOrMore[str],
        transformer: OneOrMore[TransformerLike],
        name: Optional[str] = None,
        description: Optional[str] = None,
        output: OneOrMore[str] = None,
        source: Optional[str] = None,
        options: dict = None
    ):
        self.input = input
        self.transformer = make_robust_transformer(transformer)
        self.name = name
        self.description = description
        self.output = output  # unused
        self.source = source
        self.options = options if options is not None else {}

    @cached_property
    def _init_attr_list(self):
        return list(signature(self.__init__).parameters)

    def __repr__(self):
        indent = 4
        attr_list = self._init_attr_list
        attrs_str = ',\n'.join(
            '{indent}{attr_name}={attr_val!s}'.format(
                indent=' ' * indent,
                attr_name=attr,
                attr_val=getattr(self, attr)
            ) for attr in attr_list
        )
        return '{clsname}(\n{attrs_str}\n)'.format(
            clsname=type(self).__name__, attrs_str=attrs_str)

    def as_input_transformer_tuple(
        self
    ) -> Tuple[OneOrMore[str], RobustTransformer, dict]:
        """Return an tuple for passing to DataFrameMapper constructor"""
        return self.input, self.transformer, {'alias': self.output}

    def as_feature_engineering_pipeline(
        self
    ) -> ballet.pipeline.FeatureEngineeringPipeline:
        """Return standalone FeatureEngineeringPipeline with this feature"""
        return ballet.pipeline.FeatureEngineeringPipeline(self)

    @property
    def author(self) -> Optional[str]:
        """The author of this feature if it can be inferred from its source

        The author can be inferred if the module the feature was defined in
        follows the pattern
        ``package.subpackage.user_username.feature_featurename``. Otherwise,
        returns ``None``.
        """
        if self.source:
            pieces = self.source.rsplit('.', maxsplit=2)
            if len(pieces) >= 2:
                user_str = pieces[-2]
                if user_str.startswith('user_'):
                    return user_str[len('user_'):]

        return None

    _pipeline = None

    @property
    def pipeline(self) -> ballet.pipeline.FeatureEngineeringPipeline:
        """A feature engineering pipeline containing just this feature"""
        if self._pipeline is None:
            self._pipeline = self.as_feature_engineering_pipeline()

        return self._pipeline

    def fit(self, X, y=None):
        """Fit feature.pipeline"""
        return self.pipeline.fit(X, y=y)

    def transform(self, X):
        """Transform data using feature.pipeline"""
        return self.pipeline.transform(X)

    def fit_transform(self, X, y=None):
        """Fit feature.pipeline and then transform data"""
        return self.fit(X, y=y).transform(X)
