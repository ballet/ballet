.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/ballet/ballet/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

ballet could always use more documentation, whether as part of the
official ballet docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/ballet/ballet/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `ballet` for local development.

1. Fork the `ballet` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/ballet.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development::

    $ mkvirtualenv ballet
    $ cd https://github.com/ballet/ballet/blob/master/ballet/
    $ make install-develop

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass flake8 and the
   tests, including testing other Python versions with tox::

    $ make lint
    $ make test
    $ make test-all

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.md.
3. The pull request should work for Python 3.7 and 3.8. Check
   GitHub Actions and make sure that the tests pass for all supported
   Python versions.

Tips
----

To run a subset of tests::

$ python -m pytest -k some_test


Deploying
---------

A reminder for the maintainers on how to deploy. This is a simplified release workflow that does not update the version with a dev postfix (i.e. there is no 0.1.2-dev after 0.1.1 release).

#. Make sure all your changes are committed.

#. Create an entry in HISTORY.md for this release, and stage your changes (without committing yet)::

   $ git add HISTORY.md

#. Create and tag a new release (commits all staged changes)::

   $ bumpversion minor  # i.e. 0.1.1 -> 0.2.0

#. Push changes::

   $ git push --tags origin master

GitHub Actions will then deploy to PyPI if tests pass.

Source code organization
------------------------

This is a quick overview to the Ballet core source code organization.

.. list-table::
   :width: 100%
   :header-rows: 1

   * - path
     - description
   * - `cli.py <https://github.com/ballet/ballet/blob/master/ballet/cli.py>`__
     - the `ballet` command line utility
   * - `client.py <https://github.com/ballet/ballet/blob/master/ballet/client.py>`__
     - the interactive client for users
   * - `contrib.py <https://github.com/ballet/ballet/blob/master/ballet/contrib.py>`__
     - collecting feature definitions from individual modules in source files in the file system
   * - `eng/base.py <https://github.com/ballet/ballet/blob/master/ballet/eng/base.py>`__
     - abstractions for transformers used in feature definitions, such as      as ``BaseTransformer``
   * - `eng/{misc,missing,ts}.py <https://github.com/ballet/ballet/blob/master/ballet/eng/>`__
     - custom transformers for missing data, time series problems, and more
   * - `eng/external.py <https://github.com/ballet/ballet/blob/master/ballet/eng/external>`__
     - re-export of transformers from external libraries such as      scikit-learn and feature_engine
   * - `feature.py <https://github.com/ballet/ballet/blob/master/ballet/feature.py>`__
     -  the ``Feature`` abstraction
   * - `pipeline.py <https://github.com/ballet/ballet/blob/master/ballet/pipeline.py>`__
     - the `FeatureEngineeringPipeline` abstraction
   * - `project.py <https://github.com/ballet/ballet/blob/master/ballet/project.py>`__
     - the interface between a specific Ballet project and the core Ballet library, such as utilities to load project-specific information and the `Project` abstraction
   * - `templates/ <https://github.com/ballet/ballet/blob/master/ballet/templates/>`__
     - cookiecutter templates for creating a new Ballet project or creating a new feature definition
   * - `templating.py <https://github.com/ballet/ballet/blob/master/ballet/templating.py>`__
     - user-facing functionality on top of the templates
   * - `transformer.py <https://github.com/ballet/ballet/blob/master/ballet/transformer.py>`__
     - wrappers for transformers that make them play nicely together in a pipeline
   * - `update.py <https://github.com/ballet/ballet/blob/master/ballet/update.py>`__
     - functionality to update the project template from a new upstream release
   * - `util/ <https://github.com/ballet/ballet/blob/master/ballet/util/>`__
     - various utilities
   * - `validation/main.py <https://github.com/ballet/ballet/blob/master/ballet/validation/main.py>`__
     - entry point for all validation routines
   * - `validation/base.py <https://github.com/ballet/ballet/blob/master/ballet/validation/base.py>`__
     - abstractions used in validation such as the `FeaturePerformanceEvaluator`
   * - `validation/common.py <https://github.com/ballet/ballet/blob/master/ballet/validation/common.py>`__
     - common functionality used in validation, such as the ability to collect relevant changes between a current environment and a reference environment (such as a pull request vs a default branch)
   * - `validation/entropy.py <https://github.com/ballet/ballet/blob/master/ballet/validation/entropy.py>`__
     - statistical estimation routines used in feature definition selection algorithms, such as estimators for entropy, mutual information, and conditional mutual information
   * - `validation/feature_acceptance/ <https://github.com/ballet/ballet/blob/master/ballet/validation/feature_acceptance/>`__
     - validation routines for feature acceptance
   * - `validation/feature_pruning/ <https://github.com/ballet/ballet/blob/master/ballet/validation/feature_pruning/>`__
     - validation routines for feature pruning
   * - `validation/feature_api/ <https://github.com/ballet/ballet/blob/master/ballet/validation/feature_api/>`__
     - validation routines for feature APIs
   * - `validation/project_structure/ <https://github.com/ballet/ballet/blob/master/ballet/validation/project_structure/>`__
     - validation routines for project structure
