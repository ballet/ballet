import logging
import os

import ballet
import ballet.contrib
import ballet.util.mod
import click
import numpy as np
from ballet.compat import safepath
from ballet.util.log import stacklog
from ballet.util.io import write_tabular

import {{ cookiecutter.project_slug }}
from {{ cookiecutter.project_slug }}.load_data import load_data


logger = logging.getLogger(__name__)


def get_contrib_features():
    return ballet.contrib.get_contrib_features({{cookiecutter.project_slug}})


@stacklog(logger.info, 'Building features')
def build_features(X_df):
    features = get_contrib_features()
    mapper = ballet.feature.make_mapper(features)
    X = mapper.fit_transform(X_df)
    return X, mapper


def build_features_from_dir(input_dir, return_mapper=False):
    # fit mapper on training data
    X_df_tr, _ = load_data()
    X_tr, mapper_X = build_features(X_df_tr)

    with stacklog(logger.info, 'Loading data from {}'.format(input_dir)):
        X_df, y_df = load_data(input_dir=input_dir)

    with stacklog(logger.info, 'Building features'):
        X = mapper_X.transform(X_df)

    y = np.asarray(y_df)

    if return_mapper:
        return X, y, mapper_X
    else:
        return X, y


def save_features(X, y, output_dir):
    logger.info('Saving features...')
    os.makedirs(output_dir, exist_ok=True)
    fn = os.path.join(safepath(output_dir), 'features.pkl')
    write_tabular(X, fn)
    logger.info('Saved features to {}'.format(fn))


@click.command()
@click.argument('input_dir', type=click.Path(
    exists=True, readable=True, dir_okay=True))
@click.argument('output_dir', type=click.Path())
def main(input_dir, output_dir):
    '''Build features from raw data in input_dir and save to output_dir

    Example usage::

        python -m {{ cookiecutter.project_slug }}.features.build_features /path/to/input/dir /path/to/output/dir
    '''
    X, y = build_features_from_dir(input_dir)
    save_features(X, y, output_dir)


if __name__ == '__main__':
    ballet.util.log.enable(logger=logger, level=logging.INFO)
    main()
