import logging
import os

import ballet
import ballet.contrib
import ballet.util.mod
from ballet.compat import safepath
from ballet.eng.misc import IdentityTransformer
from ballet.util.log import stacklog
from ballet.util.io import write_tabular

import {{ cookiecutter.project_slug }}
from {{ cookiecutter.project_slug }}.load_data import load_data


logger = logging.getLogger(__name__)


def get_contrib_features():
    """Get contrib features for this project

    Returns:
        List[ballet.Feature]: list of Feature objects
    """
    return ballet.contrib.get_contrib_features({{cookiecutter.project_slug}})


def get_target_encoder():
    """Get encoder for the prediction target

    Returns:
        transformer-like
    """
    return IdentityTransformer()


@stacklog(logger.info, 'Building features and target')
def build(X_df=None, y_df=None):
    """Build features and target

    Args:
        X_df (DataFrame): raw variables
        y_df (DataFrame): raw target

    Returns:
        dict with keys X_df, features, mapper_X, X, y_df, encoder_y, y
    """
    if X_df is None:
        X_df, _ = load_data()
    if y_df is None:
        _, y_df = load_data()

    features = get_contrib_features()
    mapper_X = ballet.feature.make_mapper(features)
    if features:
        X = mapper_X.fit_transform(X_df)
    else:
        X = None

    encoder_y = get_target_encoder()
    y = encoder_y.fit_transform(y_df)

    return {
        'X_df': X_df,
        'features': features,
        'mapper_X': mapper_X,
        'X': X,
        'y_df': y_df,
        'encoder_y': encoder_y,
        'y': y,
    }


def save_features(X, output_dir):
    """Save built features to output directory"""
    fn = os.path.join(safepath(output_dir), 'features.pkl')
    with stacklog(logger.info, 'Saving features to {}'.format(fn)):
        os.makedirs(output_dir, exist_ok=True)
        write_tabular(X, fn)


def save_target(y, output_dir):
    """Save built target to output directory"""
    fn = os.path.join(safepath(output_dir), 'target.pkl')
    with stacklog(logger.info, 'Saving target to {}'.format(fn)):
        os.makedirs(output_dir, exist_ok=True)
        write_tabular(y, fn)
