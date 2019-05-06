from functools import update_wrapper

import click

from ballet import __version__ as version
from ballet.compat import pathlib
from ballet.contrib import _get_contrib_feature_from_module
from ballet.exc import (
    FeatureRejected, InvalidFeatureApi, InvalidProjectStructure)
from ballet.project import Project
from ballet.util.git import CustomDiffer, get_diff_endpoints_from_commit_range
from ballet.util.log import SIMPLE_LOG_FORMAT, enable, stacklog
from ballet.util.mod import (
    import_module_from_modname, import_module_from_relpath)
from ballet.validation.common import ChangeCollector, get_accepted_features
from ballet.validation.feature_acceptance.validator import (
    GFSSFAcceptanceEvaluator)
from ballet.validation.feature_api.validator import validate_feature_api
from ballet.validation.main import prune_existing_features


@click.group()
@click.version_option(version)
def cli():
    """Command line interface for ballet projects"""
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


def enable_post_mortem(f):
    @click.pass_context
    def decorated(ctx, *args, **kwargs):
        try:
            return ctx.invoke(f, *args, **kwargs)
        except Exception as e:
            if not isinstance(e, click.ClickException):
                if ctx.obj['pdb']:
                    import pdb
                    pdb.post_mortem()
            raise
    return update_wrapper(decorated, f)


def pass_project(f):
    @click.pass_context
    def decorated(ctx, *args, **kwargs):
        project = ctx.obj['project']
        return ctx.invoke(f, project, *args, **kwargs)
    return update_wrapper(decorated, f)


@cli.group(chain=True)
@click.option('-v', '--verbose',
              count=True,
              help='Show debug output')
@click.option('--pdb/--no-pdb',
              help='Drop into debugger on error')
@click.pass_context
def validate(ctx, verbose, pdb):
    """Run individual validation checks"""
    ctx.ensure_object(dict)

    if verbose == 0:
        level = 'CRITICAL'
    elif verbose == 1:
        level = 'INFO'
    elif verbose >= 2:
        level = 'DEBUG'
    enable(level=level, format=SIMPLE_LOG_FORMAT, echo=False)

    # TODO allow project root to be specified?
    cwd = pathlib.Path.cwd()
    project = Project.from_path(cwd)
    ctx.obj['project'] = project

    ctx.obj['pdb'] = pdb


def _import_feature(feature_name, feature_path):
    # import feature
    if feature_name is not None:
        mod = import_module_from_modname(feature_name)
    elif feature_path is not None:
        cwd = pathlib.Path.cwd().resolve()
        relpath = pathlib.Path(feature_path).resolve().relative_to(cwd)
        mod = import_module_from_relpath(relpath)
    else:
        raise click.BadOptionUsage(
            'Must provide one of feature-name and feature-path')
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
@pass_project
@enable_post_mortem
@stacklog(click.echo, 'Checking feature API')
def feature_api(project, feature_name, feature_path):
    """Check Feature API

    Example usage::

       $ ballet validate feature-api --feature-name foo.bar.user_01.feature_01
    """
    feature = _import_feature(feature_name, feature_path)
    X, y = project.load_data()
    valid = validate_feature_api(feature, X, y)

    if not valid:
        raise InvalidFeatureApi


@validate.command('project-structure')
@click.option('--commit-range',
              type=str,
              help='Range of commits to diff')
@pass_project
@enable_post_mortem
@stacklog(click.echo, 'Checking project structure')
def project_structure(project, commit_range):
    """Check project structure

    Example usage::

       $ ballet validate project-structure --commit-range master...HEAD
    """

    repo = project.repo
    endpoints = get_diff_endpoints_from_commit_range(repo, commit_range)
    differ = CustomDiffer(endpoints)
    change_collector = ChangeCollector(project, differ)
    changes = change_collector.collect_changes()
    valid = not changes.inadmissible_diffs

    if not valid:
        raise InvalidProjectStructure


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
@pass_project
@enable_post_mortem
@stacklog(click.echo, 'Judging feature acceptance')
def feature_acceptance(project, feature_name, feature_path):
    """Judge feature acceptance

    Example usage::

       $ ballet validate --debug feature-acceptance --feature-name foo.bar.user_01.feature_01
    """  # noqa E401

    out = project.build()
    X_df, y, features = out['X_df'], out['y'], out['features']

    proposed_feature = _import_feature(feature_name, feature_path)
    accepted_features = get_accepted_features(features, proposed_feature)
    evaluator = GFSSFAcceptanceEvaluator(X_df, y, accepted_features)
    accepted = evaluator.judge(proposed_feature)

    if not accepted:
        raise FeatureRejected


@validate.command('feature-pruning')
@pass_project
@enable_post_mortem
@stacklog(click.echo, 'Pruning existing features')
def feature_pruning(project):
    """Evaluate feature pruning

    Example usage::

      $ ballet validate feature-pruning
    """

    prune_existing_features(project, force=True)
