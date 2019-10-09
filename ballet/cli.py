import click

from ballet import __version__ as version


@click.group()
@click.version_option(version)
def cli():
    pass


@cli.command()
def quickstart():
    """Generate a brand-new ballet project"""
    import ballet.templating
    import ballet.util.log
    ballet.util.log.enable(level='INFO',
                           format=ballet.util.log.SIMPLE_LOG_FORMAT,
                           echo=False)
    ballet.templating.render_project_template()


@cli.command('update-project-template')
@click.option('--push/--no-push', '-p',
              default=False,
              help='Push updates to remote on success')
def update_project_template(push):
    """Update an existing ballet project from the upstream template"""
    import ballet.update
    import ballet.util.log
    ballet.util.log.enable(level='INFO',
                           format=ballet.util.log.SIMPLE_LOG_FORMAT,
                           echo=False)
    ballet.update.update_project_template(push=push)


@cli.command('start-new-feature')
def start_new_feature():
    """Start working on a new feature from a template"""
    import ballet.templating
    import ballet.util.log
    ballet.util.log.enable(level='INFO',
                           format=ballet.util.log.SIMPLE_LOG_FORMAT,
                           echo=False)
    ballet.templating.start_new_feature()


@cli.command('validate')
@click.option('--check-all', '-A',
              default=False)
@click.option('--check-project-structure/--no-check-project-structure',
              envvar='BALLET_CHECK_PROJECT_STRUCTURE',
              default=False)
@click.option('--check-feature-api/--no-check-feature-api',
              envvar='BALLET_CHECK_FEATURE_API',
              default=False)
@click.option('--evaluate-feature-acceptance/--no-evaluate-feature-acceptance',
              envvar='BALLET_EVALUATE_FEATURE_ACCEPTANCE',
              default=False)
@click.option('--evaluate-feature-pruning/--no-evaluate-feature-pruning',
              envvar='BALLET_EVALUATE_FEATURE_PRUNING',
              default=False)
def validate(check_all, check_project_structure, check_feature_api,
             evaluate_feature_acceptance, evaluate_feature_pruning):
    """Validate a project changes from a branch"""
    import pathlib

    import ballet.util.log
    import ballet.project
    import ballet.validation.main

    ballet.util.log.enable(level='DEBUG',
                           format=ballet.util.log.SIMPLE_LOG_FORMAT,
                           echo=False)
    cwd = pathlib.Path.cwd()
    project = ballet.project.Project.from_path(cwd)
    ballet.validation.main.validate(
        project,
        check_project_structure or check_all,
        check_feature_api or check_all,
        evaluate_feature_acceptance or check_all,
        evaluate_feature_pruning or check_all)
