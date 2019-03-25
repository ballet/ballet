import click

import ballet


@click.group()
@click.version_option(ballet.__version__)
def cli():
    pass


@cli.command()
def quickstart():
    """Generate a brand-new ballet project"""
    import ballet.templating
    ballet.templating.render_project_template()


@cli.command('update-project-template')
@click.option('--push/--no-push', '-p',
              default=False,
              help='Push updates to remote on success')
def update_project_template(push):
    """Update an existing ballet project from the upstream template"""
    import ballet.update
    ballet.update.main(push=push)


@cli.command('start-new-feature')
def start_new_feature():
    """Start working on a new feature from a template"""
    import ballet.templating
    ballet.templating.start_new_feature()
