from typing import Dict, List, Optional

import funcy as fy
import numpy as np
import pandas as pd

import ballet
from ballet.transformer import get_transformer_primitives
from ballet.util import asarray2d
from ballet.validation.entropy import (
    estimate_conditional_information, estimate_mutual_information,)


def countunique(z: np.ndarray, axis=0):
    return np.apply_along_axis(
        lambda arr: len(np.unique(arr)), axis, z)


@fy.memoize(key_func=lambda feature, values, y: id(feature))
def _summarize_feature(
    feature: 'ballet.feature.Feature',
    values: Optional[Dict['ballet.feature.Feature', Optional[np.ndarray]]],
    y: Optional[np.ndarray],
) -> dict:
    """Summarize a single feature"""
    result = {
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
        'mutual_information': np.nan,
        'conditional_mutual_information': np.nan,
        'mean': np.nan,
        'std': np.nan,
        'variance': np.nan,
        'nunique': np.nan,
    }

    # if feature values are missing here, the values are left at nans from
    # above
    if values is not None and y is not None:
        z = values[feature]
        if z is not None:
            feature_values_list = [
                feature_values
                for other_feature, feature_values in values.items()
                if other_feature is not feature and feature_values is not None
            ]
            if feature_values_list:
                x = np.concatenate(feature_values_list, axis=1)
            else:
                x = np.empty((z.shape[0], 0))

            result['mutual_information'] = estimate_mutual_information(z, y)
            result['conditional_mutual_information'] = \
                estimate_conditional_information(z, y, x)
            result['mean'] = np.mean(np.mean(z, axis=0))  # same thing anyway
            result['std'] = np.mean(np.std(z, axis=0))
            result['variance'] = np.mean(np.var(z, axis=0))
            result['nunique'] = np.mean(countunique(z, axis=0))

    return result


def discover(
    features: List['ballet.feature.Feature'],
    X_df: Optional[pd.DataFrame],
    y_df: Optional[pd.DataFrame],
    y: Optional[np.ndarray],
    input: Optional[str] = None,
    primitive: Optional[str] = None
) -> pd.DataFrame:
    """Discover existing features

    Display information about existing features including summary statistics on
    the development dataset.  If the feature extracts multiple feature values,
    then the summary statistics (e.g. mean, std, nunique) are computed for each
    feature value and then averaged. If the development dataset cannot be
    loaded, computation of summary statistics is skipped.

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
    records = []

    if X_df is not None and y_df is not None and y is not None:

        @fy.ignore(Exception)
        def get_feature_values(feature):
            return asarray2d(
                feature
                .as_feature_engineering_pipeline()
                .fit_transform(X_df, y_df))

        values = {
            feature: get_feature_values(feature)
            for feature in features
        }
        y = asarray2d(y)
        summarize = fy.rpartial(_summarize_feature, values, y)

    else:
        summarize = fy.rpartial(_summarize_feature, None, None)

    for feature in features:
        if input and input not in feature.input and input != feature.input:
            continue
        if (
            primitive
            and primitive not in get_transformer_primitives(
                feature.transformer)
        ):
            continue
        summary = summarize(feature)
        records.append(summary)

    return pd.DataFrame.from_records(records)
