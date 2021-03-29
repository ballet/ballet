import click
from stacklog import stacklog

from ballet import __version__ as version


@click.group()
@click.version_option(version)
@click.option('-v', '--verbose',
              count=True,
              help='Increase verbosity')
@click.option('-q', '--quiet',
              count=True,
              help='Decrease verbosity'
              )
def cli(verbose, quiet):
    """The Ballet command line interface"""
    import ballet.util.log

    # Process logging
    count = verbose - quiet
    if count <= -1:
        level = 'CRITICAL'
    elif count == 0:
        level = 'INFO'
    else:
        level = 'DEBUG'
    ballet.util.log.enable(level=level,
                           format=ballet.util.log.DETAIL_LOG_FORMAT,
                           echo=False)


@cli.command()
@click.option('--create-github-repo',
              is_flag=True,
              default=False,
              help='Create repo on GitHub for resulting project')
@click.option('--github-token',
              envvar='GITHUB_TOKEN',
              default=None,
              help='github personal access token')
@stacklog(click.echo, 'Generating new ballet project')
def quickstart(create_github_repo, github_token):
    """Generate a brand-new ballet project"""
    import ballet.templating
    ballet.templating.render_project_template(
        create_github_repo=create_github_repo,
        github_token=github_token,

    )


@cli.command('update-project-template')
@click.option('--push/--no-push', '-p',
              default=False,
              help='Push updates to remote on success')
@click.option('--path', 'project_template_path',
              default=None,
              help='Specify override for project template path '
                   '(i.e. gh:ballet/ballet)')
@stacklog(click.echo, 'Updating project template')
def update_project_template(push, project_template_path):
    """Update an existing ballet project from the upstream template"""
    import ballet.update
    ballet.update.update_project_template(
        push=push, project_template_path=project_template_path)


@cli.command('start-new-feature')
@click.option('--branching/--no-branching',
              default=True,
              help='whether to create a new branch to work on the feature')
@stacklog(click.echo, 'Starting new feature')
def start_new_feature(branching):
    """Start working on a new feature from a template"""
    import ballet.templating
    ballet.templating.start_new_feature(branching=branching)


@cli.command('validate')
@click.option('--check-all', '-A',
              is_flag=True,
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
@stacklog(click.echo, 'Validating project')
def validate(check_all, check_project_structure, check_feature_api,
             evaluate_feature_acceptance, evaluate_feature_pruning):
    """Validate project changes from a candidate branch"""
    import ballet.project
    import ballet.util.log
    import ballet.validation.main

    # over-write logging settings?
    ballet.util.log.enable(level='DEBUG',
                           format=ballet.util.log.DETAIL_LOG_FORMAT,
                           echo=False)

    project = ballet.project.Project.from_cwd()
    ballet.validation.main.validate(
        project,
        check_project_structure or check_all,
        check_feature_api or check_all,
        evaluate_feature_acceptance or check_all,
        evaluate_feature_pruning or check_all)
