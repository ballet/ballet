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
@click.argument('input_dir', type=click.Path(
    exists=True, file_okay=False, readable=True))
@click.argument('output_dir', type=click.Path(file_okay=False))
def engineer_features(input_dir, output_dir):
    """Engineer features"""
    X_df, y_df = api.load_data(input_dir=input_dir)
    result = api.engineer_features()
    pipeline, encoder = result.pipeline, result.encoder

    X_ft = pipeline.transform(X_df, y=y_df)
    y_ft = encoder.transform(y_df)

    save_features(X_ft, output_dir)
    save_targets(y_ft, output_dir)


if __name__ == '__main__':
    cli()
