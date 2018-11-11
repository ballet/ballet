=====
Usage
=====

Ballet projects
---------------

In this section, we will describe how the Ballet framework can be leveraged for a hypothetical
collaborative feature engineering project, which we will call ballet-project.

Before creating ballet-project, the maintainer must have a training dataset used for developing features and details about
the prediction problem they are ultimately trying to solve.

Project instantiation
~~~~~~~~~~~~~~~~~~~~~

To instantiate a project, use the `ballet-quickstart` command:

.. code-block:: console

    $ ballet-quickstart
    full_name [Your Name]: Jane Developer
    email [you@example.com]: jane@developer.org
    github_username [jane]: jane_developer
    project_name [Predict Foo]: Predict house prices
    project_slug [predict_house_prices]: ballet_project
    Select problem_type:
    1 - classification
    2 - regression
    Choose from 1, 2 (1, 2) [1]: 2
    Select classification_type:
    1 - n/a
    2 - binary
    3 - multiclass
    Choose from 1, 2, 3 (1, 2, 3) [1]: 1
    Select classification_scorer:
    1 - n/a
    2 - accuracy
    3 - balanced_accuracy
    4 - average_precision
    5 - brier_score_loss
    6 - f1
    7 - f1_micro
    8 - f1_macro
    9 - f1_weighted
    10 - f1_samples
    11 - neg_log_loss
    12 - precision
    13 - precision_micro
    14 - precision_macro
    15 - precision_weighted
    16 - precision_samples
    17 - recall
    18 - recall_micro
    19 - recall_macro
    20 - recall_weighted
    21 - recall_samples
    22 - roc_auc
    Choose from 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22) [1]: 1
    Select regression_scorer:
    1 - n/a
    2 - explained_variance
    3 - neg_mean_absolute_error
    4 - neg_mean_squared_error
    5 - neg_mean_squared_log_error
    6 - neg_median_absolute_error
    7 - r2
    Choose from 1, 2, 3, 4, 5, 6, 7 (1, 2, 3, 4, 5, 6, 7) [1]: 5
    Initialized empty Git repository in /tmp/ballet_project/.git/
    [master (root-commit) 95a70e9] Automatically generated files from ballet-quickstart
     11 files changed, 252 insertions(+)
     create mode 100644 .gitignore
     create mode 100644 .travis.yml
     create mode 100644 README.md
     create mode 100644 ballet.yml
     create mode 100644 ballet_project/__init__.py
     create mode 100644 ballet_project/conf.py
     create mode 100644 ballet_project/features/__init__.py
     create mode 100644 ballet_project/features/contrib/__init__.py
     create mode 100644 ballet_project/load_data.py
     create mode 100644 requirements.txt
     create mode 100755 validate.py


This command uses `cookiecutter`_ to render a templated project using information supplied by the user. The resulting
files are then committed to a new git repository. Note that the specification of a scorer for the not-chosen problem
type can be skipped (by selecting ``n/a``).

Let's see what files have been created:

* ``.gitignore``: a reference gitignore file.
* ``.travis.yml``: a `Travis CI`_ configuration file pre-configured to run a ballet validation suite.
* ``README.md``: a basic README for your project.
* ``ballet.yml``: a Ballet configuration file, with details about the prediction problem, the training data, and
    location of feature engineering source code.
* ballet_project: a Python package that implements an empty feature engineering pipeline.
* ``requirements.txt``: package dependencies
* ``validate.py``: a driver script for the validation suite that calls out to Ballet for most functionality.

Developing new features
~~~~~~~~~~~~~~~~~~~~~~~

A contributors wants to develop a new feature for inclusion in the project. First, they fork the project and create a
new Python subpackage under the "contrib" directory. In this example, the contrib directory is
``ballet_project/features/contrib``, but it can be changed in ``ballet.yml``.

* The new subpackage must be named like ``user_<github username>``.
* The new submodule that will contain the feature must be named like ``feature_<feature name>.py``.

Within the feature submodule, the contributor can write arbitrary Python code. Ultimately, a single object that is an
instance of ``ballet.Feature`` must be defined; it will be imported by the feature engineering pipeline.

.. code-block:: python

   import ballet.eng
   from ballet import Feature

   input = 'A'
   transformer = ballet.eng.misc.IdentityTransformer()
   feature = Feature(input=input, transformer=transformer)

In this example, a feature is defined that will receive column ``'A'`` from the data and passes it through unmodified.

The contributor now commits their changes and creates a `pull request`_ to the ``ballet_project`` repository.

Validating features
~~~~~~~~~~~~~~~~~~~

The ``ballet_project`` repository has received a new pull request which triggers an automatic evaluation.

1. The PR is downloaded by an external continuous integration service, `Travis CI`_.
2. The ``validate.py`` script is run, which validates the proposed feature contribution using functionality within
the ``ballet.validation`` module.
3. If the feature can be validated successfully, the PR passes, and the proposed feature can be merged into the project.

.. _cookiecutter: https://cookiecutter.readthedocs.io/en/latest
.. _`Travis CI`: https://travis-ci.org
.. _`pull request`: https://help.github.com/articles/about-pull-requests/
