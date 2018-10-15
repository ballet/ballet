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
    return ballet.contrib.get_contrib_features({{cookiecutter.project_slug}})


def get_target_encoder():
    return IdentityTransformer()


@stacklog(logger.info, 'Building features and target')
def build(X_df=None, y_df=None, return_mappers=False):
    if X_df is None:
        X_df, _ = load_data()
    if y_df is None:
        _, y_df = load_data()

    features = get_contrib_features()
    mapper_X = ballet.feature.make_mapper(features)
    X = mapper_X.fit_transform(X_df)

    encoder_y = get_target_encoder()
    y = encoder_y.fit_transform(y_df)

    if return_mappers:
        return X, y, mapper_X, encoder_y
    else:
        return X, y


def save_features(X, output_dir):
    fn = os.path.join(safepath(output_dir), 'features.pkl')
    with stacklog(logger.info, 'Saving features to {}'.format(fn)):
        os.makedirs(output_dir, exist_ok=True)
        write_tabular(X, fn)


def save_target(y, output_dir):
    fn = os.path.join(safepath(output_dir), 'target.pkl')
    with stacklog(logger.info, 'Saving target to {}'.format(fn)):
        os.makedirs(output_dir, exist_ok=True)
        write_tabular(y, fn)
