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


@click.group(chain=True)
@click.option('--debug/--no-debug',
              help='Show debug output')
@click.pass_context
def validate(ctx, debug):
    """Run individual validation checks"""
    ctx.ensure_object(dict)

    from ballet.util.log import SIMPLE_LOG_FORMAT, enable, logger
    if debug:
        level = 'DEBUG'
    else:
        level = 'INFO'
    enable(level=level, format=SIMPLE_LOG_FORMAT, echo=False)

    from ballet.compat import pathlib
    from ballet.project import Project

    # TODO allow project root to be specified?
    cwd = pathlib.Path.cwd()
    project = Project.from_path(cwd)
    ctx.obj['project'] = project


cli.add_command(validate)


def _import_feature(feature_name, feature_path):
    from ballet.compat import pathlib
    from ballet.contrib import _get_contrib_feature_from_module
    from ballet.util.mod import (
        import_module_from_modname, import_module_from_relpath)

    # import feature
    if feature_name is not None and feature_path is None:
        mod = import_module_from_modname(feature_name)
    elif feature_path is not None and feature_name is None:
        cwd = pathlib.Path.cwd().resolve()
        relpath = pathlib.Path(feature_path).resolve().relative_to(cwd)
        mod = import_module_from_relpath(relpath)
    else:
        raise click.BadOptionUsage('Exactly one of feature-name and '
                                   'feature-path should be specified')
    return _get_contrib_feature_from_module(mod)


@click.option('--feature-name',
              type=str,
              help='Feature module name')
@click.option('--feature-path',
              type=click.Path(exists=True,
                              file_okay=True,
                              dir_okay=False,
                              readable=True),
              help='Relative path to feature module')
@click.option('-v', '--verbose',
              is_flag=True,
              help='Show debug output')
def validate(check_feature_api, feature_name, feature_path, verbose):
    """Run individual validation checks"""
    from ballet.util.log import SIMPLE_LOG_FORMAT, enable, logger
    level = 'INFO' if not verbose else 'DEBUG'
    enable(level=level, format=SIMPLE_LOG_FORMAT, echo=False)

    from ballet.compat import pathlib
    from ballet.contrib import _get_contrib_feature_from_module
    from ballet.project import Project
    from ballet.util.mod import (
        import_module_from_modname, import_module_from_relpath)
    from ballet.validation.feature_api.validator import validate_feature_api

    # TODO allow project root to be specified?
    cwd = pathlib.Path.cwd()
    project = Project.from_path(cwd)

    if check_feature_api:

        # import feature
        if feature_name is not None and feature_path is None:
            mod = import_module_from_modname(feature_name)
        elif feature_path is not None and feature_name is None:
            relpath = pathlib.Path(feature_path).relative_to(cwd)
            mod = import_module_from_relpath(relpath)
        else:
            raise click.BadOptionUsage('Exactly one of feature-name and '
                                       'feature-path should be specified')
        feature = _get_contrib_feature_from_module(mod)

        # load data
        X, y = project.load_data()

        # validate!
        validate_feature_api(feature, X, y)

        logger.info('Check feature API successful.')
