import click

import ballet


@click.group()
@click.version_option(ballet.__version__)
def cli():
    pass


@cli.command()
def quickstart():
    """Generate a brand-new ballet project"""
    import ballet.quickstart
    ballet.quickstart.main()


@cli.command('update-project-template')
def update_project_template():
    """Update an existing ballet project from the upstream template"""
    import ballet.update
    ballet.update.main()
