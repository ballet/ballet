import logging

import ballet
import ballet.contrib
import ballet.util.mod
import click
from ballet.eng.misc import IdentityTransformer
from ballet.pipeline import FeatureEngineeringPipeline
from ballet.util.io import save_features, save_targets
from ballet.util.log import stacklog

import {{ cookiecutter.package_slug }}
from {{ cookiecutter.package_slug }}.load_data import load_data


logger = logging.getLogger(__name__)


def collect_contrib_features():
    """Get contrib features for this project

    Returns:
        List[ballet.Feature]: list of Feature objects
    """
    return ballet.contrib.collect_contrib_features({{ cookiecutter.package_slug }})


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

    features = collect_contrib_features()
    mapper_X = FeatureEngineeringPipeline(features)
    X = mapper_X.fit_transform(X_df)

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


@click.command()
@click.argument('input_dir', type=click.Path(
    exists=True, file_okay=False, readable=True))
@click.argument('output_dir', type=click.Path(file_okay=False))
def main(input_dir, output_dir):
    """Engineer features"""

    import ballet.util.log
    ballet.util.log.enable(logger=logger, level='INFO', echo=False)
    ballet.util.log.enable(logger=ballet.util.log.logger, level='INFO',
                           echo=False)

    X_df, y_df = load_data(input_dir=input_dir)
    out = build()

    mapper_X = out['mapper_X']
    encoder_y = out['encoder_y']

    X_ft = mapper_X.transform(X_df)
    y_ft = encoder_y.transform(y_df)

    save_features(X_ft, output_dir)
    save_targets(y_ft, output_dir)


if __name__ == '__main__':
    main()
