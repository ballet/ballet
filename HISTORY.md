# History

## 0.8.1 (2020-02-16)

* Fix bug with `str(t)` or `repr(t)` for `SimpleFunctionTransformer`

## 0.8.0 (2021-02-02)

* Fix bug with detecting updates to Ballet due to PyPI API outage
* Fix some dependency conflicts
* Reference ballet-assemble in project template
* Bump feature_engine to 1.0

## 0.7.11 (2020-09-16)

* Reduce verbosity of conversion approach logging by moving some messages to TRACE level
* Implement "else" transformer for `ConditionalTransformer`
* Improve GFSSF iteration logging

## 0.7.10 (2020-09-08)

* Fix bug with different treatment of y_df and y; now, y_df is passed to the feature engineering pipeline, and y is passed to the feature validation routines as applicable.
* Switch back to using Gitter

## 0.7.9 (2020-08-15)

* Add give_advice feature for FeatureAPICheck and other checks to log message on how to fix failure
* Improve logging of GFSSFAccepter and GFSSFPruner
* Improve `__str__` for DelegatingRobustTransformer and consequently consumers
* Change default log format to SIMPLE_LOG_FORMAT
* Various bug fixes and improvements

## 0.7.8 (2020-08-13)

* Add CanTransformNewRowsCheck to feature API checks

## 0.7.7 (2020-08-12)

* Support `None` as the transformer in a `Feature`, it will be automatically converted to an `IdentityTransformer`
* Implement `ColumnSelector`
* Update docs
* Various bug fixes and improvements

## 0.7.6 (2020-08-12)

* Re-export feature engineering primitives from various libraries
* Show type annotations in docs
* Update guides
* Various bug fixes and improvements

## 0.7.5 (2020-08-03)

* Make validator parameters configurable in ballet.yml file (e.g. λ_1 and λ_2 for GFSSF algorithms)
* Support dynaconf 3.x

## 0.7.4 (2020-07-22)

* Accept logger names, as well as logger instances, in `ballet.util.log.enable`
* Updated docs

## 0.7.3 (2020-07-21)

* Add `load_data` method with built-in caching to project API
* Fix bug in GFSSF accepter
* Always use encoded target during validation
* Various bug fixes and improvements

## 0.7.2 (2020-07-21)

* Add sample analysis notebook to project template
* Add binder url/badge to project template
* Fix bug with enabling logging with multiple loggers

## 0.7.1 (2020-07-20)

* Add client for easy interactive usage (`ballet.b`)
* Add binder setup to project template

## 0.7 (2020-07-17)

* Revamp project template: update project structure, create single API via FeatureEngineeringProject, use and add support for pyinvoke, revamp build into engineer_features, support repolockr bot
* Improve ballet.project.Project: can create by ascending from given path, can create from current working directory, can resolve arbitrary project symbol, exposes project's API
* Check for and notify of new release of ballet during project update (`ballet update-project-template`)
* Add ComputedValueTransformer to ballet.eng
* Move stacklog to separate project and install it
* Add validators that {never,always} accept submissions
* Add feature API checks to ensure that the feature can fit and transform a single row
* Add feature engineering guide to documentation and significantly expand contributor guide
* Add bot installation instructions to maintainer guide
* Add type annotations throughout
* Drop support for py35, add support for py38
* Deprecate modeling code
* Various bug fixes and improvements

## 0.6 (2019-11-12)

* Implement GFSSF validators and random validators
* Improve validators and allow validators to be configured in ballet.yml
* Improve project template
* Create ballet CLI
* Bug fixes and performance improvements

## 0.5 (2018-10-14)

* Add project template and ballet-quickstart command
* Add project structure checks and feature API checks
* Implement multi-stage validation routine driver

## 0.4 (2018-09-21)

* Implement `Modeler` for versatile modeling and evaluation
* Change project name

## 0.3 (2018-04-28)

* Implement `PullRequestFeatureValidator`
* Add `util.travis`, `util.modutil`, `util.git` util modules

## 0.2

* Implement `ArrayLikeEqualityTestingMixin`
* Implement `collect_contrib_features`

## 0.1

* First release on PyPI
