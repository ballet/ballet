from inspect import signature

from funcy import cached_property

import ballet.pipeline
from ballet.transformer import make_robust_transformer

__all__ = ('Feature', )


class Feature:
    """A logical feature

    Conceptually, a logical feature is a learned function that maps raw
    variables in one data instance to a vector of feature values. A logical
    feature can produce either a scalar feature value for each instance or a
    vector of feature values, as in the case of an embedding technique like PCA
    or the one-hot encoding of a categorical variable.

    Args:
        input (str, Collection[str]): required columns from the input
            dataframe needed for the transformation
        transformer (transformer-like, List[transformer-like]): transformer, or
            sequence of transformers. A "transformer" is an instance of a class
            that provides a fit/transform-style learned transformation.
            Alternately, a callable can be provided, either by itself or in a
            list, in which case it will be converted into a
            ``SimpleFunctionTransformer`` for convenience.
        name (str, optional): name of the feature
        description (str, optional): description of the feature
        output (str, list[str]): ordered sequence of names of features
            produced by this transformer
        source (path-like): the source file in which this feature was defined
        options (dict): options
    """

    def __init__(self, input, transformer, name=None, description=None,
                 output=None, source=None, options=None):
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
        attr_list = self._init_attr_list
        attrs_str = ', '.join(
            '{attr_name}={attr_val!r}'.format(
                attr_name=attr, attr_val=getattr(self, attr)
            ) for attr in attr_list
        )
        return '{clsname}({attrs_str})'.format(
            clsname=type(self).__name__, attrs_str=attrs_str)

    def as_input_transformer_tuple(self):
        """Return an tuple for passing to DataFrameMapper constructor"""
        return self.input, self.transformer, {'alias': self.output}

    def as_feature_engineering_pipeline(self):
        """Return standalone FeatureEngineeringPipeline with this feature"""
        return ballet.pipeline.FeatureEngineeringPipeline(self)
