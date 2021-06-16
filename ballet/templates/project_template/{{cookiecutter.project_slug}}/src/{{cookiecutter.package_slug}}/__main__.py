import logging

from ballet.util.io import save_features, save_targets
import click

from {{ cookiecutter.package_slug }}.api import api

logger = logging.getLogger(__name__)


@click.group()
def cli():
    import ballet.util.log
    ballet.util.log.enable(logger=logger, level='INFO', echo=False)
    ballet.util.log.enable(logger=ballet.util.log.logger, level='INFO',
                           echo=False)


@cli.command()
@click.option(
    '--train-dir',
    type=click.Path(exists=True, file_okay=False, readable=True),
    default=None)
@click.argument(
    'input_dir',
    type=click.Path(exists=True, file_okay=False, readable=True))
@click.argument(
    'output_dir',
    type=click.Path(file_okay=False))
def engineer_features(input_dir, output_dir, train_dir):
    """Engineer features"""
    # load pipeline trained on development set
    if train_dir is not None:
        X_df_tr, y_df_tr = api.load_data(input_dir=train_dir)
        result = api.engineer_features(X_df_tr, y_df_tr)
    else:
        result = api.engineer_features()
    pipeline, encoder = result.pipeline, result.encoder

    # load input data
    X_df, y_df = api.load_data(input_dir=input_dir)

    # transform entities and targets
    X_ft = pipeline.transform(X_df)
    y_ft = encoder.transform(y_df)

    # save to output dir
    save_features(X_ft, output_dir)
    save_targets(y_ft, output_dir)


if __name__ == '__main__':
    cli(prog_name='{{ cookiecutter.package_slug }}')
