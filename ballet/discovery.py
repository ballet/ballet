import funcy as fy
import numpy as np
import pandas as pd

from ballet.transformer import get_transformer_primitives
from ballet.util import asarray2d
from ballet.validation.entropy import (
    estimate_conditional_information, estimate_mutual_information,)


def countunique(z, axis=0):
    return np.apply_along_axis(
        lambda arr: len(np.unique(arr)), axis, z)


@fy.memoize(key_func=lambda feature, values, y: id(feature))
def _summarize_feature(feature, values, y) -> dict:
    z = values[feature]

    feature_values_list = [
        feature_values
        for other_feature, feature_values in values.items()
        if other_feature is not feature
    ]
    if feature_values_list:
        x = np.concatenate(feature_values_list, axis=1)
    else:
        x = np.empty((z.shape[0], 0))

    mutual_information = estimate_mutual_information(z, y)
    conditional_mutual_information = \
        estimate_conditional_information(z, y, x)
    mean = np.mean(z, axis=0)
    std = np.std(z, axis=0)
    variance = np.var(z, axis=0)
    nunique = countunique(z, axis=0)
    return {
        'name': feature.name,
        'description': feature.description,
        'input':
            [feature.input]
            if isinstance(feature.input, str)
            else feature.input,
        'transformer': feature.transformer,
        'primitives': get_transformer_primitives(feature.transformer),
        'output': feature.output,
        'author': feature.author,
        'source': feature.source,
        'mutual_information': mutual_information,
        'conditional_mutual_information':
            conditional_mutual_information,
        'mean': np.mean(mean),  # same as mean over flattened anyways
        'std': np.mean(std),
        'variance': np.mean(variance),
        'nunique': np.mean(nunique),
    }


def discover(
    features, X_df, y_df, y, input=None, primitive=None
) -> pd.DataFrame:
    """Discover existing features

    Display information about existing features including summary statistics.
    If the feature extracts multiple feature values, then the summary
    statistics (e.g. mean, std, nunique) are computed for each feature value
    and then averaged.

    The following information is shown:
    - name: the name of the feature
    - description: the description of the feature
    - input: the variables that are used as input to the feature
    - transformer: the transformer/transformer pipeline
    - output: the output columns of the feature (not usually specified)
    - author: the GitHub username of the feature's author
    - source: the fully-qualified name of the Python module that contains the
        feature
    - mutual_information: estimated mutual information between the feature (or
        averaged over feature values) and the target on the development
        dataset split
    - conditional_mutual_information: estimated conditional mutual information
        between the feature (or averaged over feature values) and the target
        conditional on all other features on the development dataset split
    - mean: mean of the feature on the development dataset split
    - std: standard deviation of the feature (or averaged over feature values)
        on the development dataset split
    - var: variance of the feature (or averaged over feature values) on the
        development dataset split
    - nunique: number of unique values of the feature (or averaged over
        feature values) on the development dataset split

    The following query operators are supported:
    - input (str): filter to only features that have ``input`` in their input/
        list of inputs
    - primitive (str): filter to only features that use primitive
        ``primitive`` (i.e. a class with name ``primitive``) in the
        transformer/transformer pipeline

    For other queries, you should just use normal DataFrame indexing::

       >>> features_df[features_df['author'] == 'jane']
       >>> features_df[features_df['name'].str.contains('married')]
       >>> features_df[features_df['mutual_information'] > 0.05]
       >>> features_df[features_df['input'].apply(
               lambda input: 'A' in input and 'B' in input)]

    Returns:
        data frame with features on the row index and columns as described
        above
    """
    y = asarray2d(y)
    records = []
    values = {
        feature: asarray2d(
            feature
            .as_feature_engineering_pipeline()
            .fit_transform(X_df, y_df)
        )
        for feature in features
    }
    for feature in features:
        if input and input not in feature.input and input != feature.input:
            continue
        if (
            primitive
            and primitive not in get_transformer_primitives(
                feature.transformer)
        ):
            continue
        summary = _summarize_feature(feature, values, y)
        records.append(summary)

    return pd.DataFrame.from_records(records)
