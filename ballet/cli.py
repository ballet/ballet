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


@validate.command('feature-api')
@click.option('--feature-name',
              type=str,
              help='Feature module name')
@click.option('--feature-path',
              type=click.Path(exists=True,
                              file_okay=True,
                              dir_okay=False,
                              readable=True),
              help='Relative path to feature module')
@click.pass_context
def feature_api(ctx, feature_name, feature_path):
    """Check Feature API"""
    project = ctx.obj['project']

    from ballet.util.log import logger
    from ballet.validation.feature_api.validator import validate_feature_api

    feature = _import_feature(feature_name, feature_path)
    X, y = project.load_data()
    valid = validate_feature_api(feature, X, y)
    if valid:
        logger.info('SUCCESS')
    else:
        logger.info('FAILURE')

    return valid


@validate.command('project-structure')
@click.option('--commit-range',
              type=str,
              help='Range of commits to diff')
@click.pass_context
def project_structure(ctx, commit_range):
    """Check project structure"""
    project = ctx.obj['project']
    from ballet.util.git import (
        CustomDiffer, get_diff_endpoints_from_commit_range)
    from ballet.util.log import logger
    from ballet.validation.common import ChangeCollector

    repo = project.repo
    endpoints = get_diff_endpoints_from_commit_range(repo, commit_range)
    differ = CustomDiffer(endpoints)
    change_collector = ChangeCollector(project, differ)
    changes = change_collector.collect_changes()
    valid = not changes.inadmissible_diffs
    if valid:
        logger.info('SUCCESS')
    else:
        logger.info('FAILURE')

    return valid


@validate.command('feature-acceptance')
@click.option('--feature-name',
              type=str,
              help='Feature module name')
@click.option('--feature-path',
              type=click.Path(exists=True,
                              file_okay=True,
                              dir_okay=False,
                              readable=True),
              help='Relative path to feature module')
@click.pass_context
def feature_acceptance(ctx, feature_name, feature_path):
    project = ctx.obj['project']

    from ballet.util.log import logger
    from ballet.validation.common import get_accepted_features
    from ballet.validation.feature_acceptance.validator import (
        GFSSFAcceptanceEvaluator)

    out = project.build()
    X_df, y, features = out['X_df'], out['y'], out['features']

    proposed_feature = _import_feature(feature_name, feature_path)
    accepted_features = get_accepted_features(features, proposed_feature)
    evaluator = GFSSFAcceptanceEvaluator(X_df, y, accepted_features)
    accepted = evaluator.judge(proposed_feature)

    if accepted:
        logger.info('ACCEPTED')
    else:
        logger.info('REJECTED')

    return accepted
