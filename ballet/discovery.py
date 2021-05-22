from ballet.util import asarray2d
import numpy as np
import pandas as pd

from ballet.validation.entropy import (
    estimate_conditional_information, estimate_mutual_information,)


def countunique(z, axis=0):
    return np.apply_along_axis(
        lambda arr: len(np.unique(arr)), axis, z)


def discover(features, X_df, y_df, y) -> pd.DataFrame:
    """Discover existing features

    Display information about existing features including summary statistics.
    If the feature extracts multiple feature values, then the summary
    statistics (e.g. mean, std, nunique) are computed for each feature value
    and then averaged.

    Returns:
        data frame with features on the row index and the following
        columns: ``name``, ``description``, ``input``, ``transformer``,
        ``output``, ``author``, ``source``, ``mutual_information``,
        ``conditional_mutual_information``, ``mean``, ``std``, ``var``,
        ``nunique``.
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
        records.append({
            'name': feature.name,
            'description': feature.description,
            'input': feature.input,
            'transformer': feature.transformer,
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
        })

    return pd.DataFrame.from_records(records)
