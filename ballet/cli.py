import click

from ballet import __version__ as version
import ballet.util.log


@click.group()
@click.version_option(version)
def cli():
    pass


@cli.command()
def quickstart():
    """Generate a brand-new ballet project"""
    import ballet.templating
    ballet.util.log.enable(level='INFO', echo=False)
    ballet.templating.render_project_template()


@cli.command('update-project-template')
@click.option('--push/--no-push', '-p',
              default=False,
              help='Push updates to remote on success')
def update_project_template(push):
    """Update an existing ballet project from the upstream template"""
    import ballet.update
    ballet.util.log.enable(level='INFO', echo=False)
    ballet.update.update_project_template(push=push)


@cli.command('start-new-feature')
def start_new_feature():
    """Start working on a new feature from a template"""
    import ballet.templating
    ballet.util.log.enable(level='INFO', echo=False)
    ballet.templating.start_new_feature()
