================
Maintainer Guide
================

Creating a Ballet project
-------------------------

You and your hundred smartest colleagues want to collaborate on a feature engineering project. How
will you organize your work? You are in the right place to learn. With the Ballet framework,
contributors to your project will write self-contained feature engineering source code. Then,
Ballet will take care of the rest: submitting proposed features as pull requests to your GitHub
repository, carefully validating the proposed features, and combining all of the accepted features
into a single feature engineering pipeline.

In this section, we will describe how the Ballet framework can be leveraged for your project, which
we will call ``myproject``.

Prerequisites
~~~~~~~~~~~~~

Before creating the project, the maintainer must have a training dataset used for developing
features and details about the prediction problem they are ultimately trying to solve.

Then, :doc:`install Ballet </installation>` on your development machine.

Project instantiation
~~~~~~~~~~~~~~~~~~~~~

To instantiate a project, use the ``ballet quickstart`` command:

.. code-block:: console

   $ ballet quickstart
   Generating new ballet project...
   full_name [Your Name]: Jane Developer
   email [you@example.com]: jane@developer.org
   github_owner [jane]: jane_developer
   project_name [Predict X]: Predict my thing
   project_slug [ballet-predict-my-thing]: ballet-my-project
   package_slug [ballet_predict_my_thing]: myproject
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
   Select pruning_action:
   1 - no_action
   2 - make_pull_request
   3 - commit_to_master
   Choose from 1, 2, 3 (1, 2, 3) [1]: 3
   Select auto_merge_accepted_features:
   1 - no
   2 - yes
   Choose from 1, 2 (1, 2) [1]: 2
   Select auto_close_rejected_features:
   1 - no
   2 - yes
   Choose from 1, 2 (1, 2) [1]: 2
   Generating new ballet project...DONE

This command uses `cookiecutter`_ to render a project template using information supplied by the
project maintainer. The resulting files are then committed to a new git repository. Note that the
specification of a scorer for the not-chosen problem type can be skipped (by selecting ``n/a``).

Let's see what files have we have created:

.. code-block:: console

   $ tree -a ballet-my-project -I .git
   ballet-my-project
   ├── .cookiecutter_context.json
   ├── .github
   │   └── repolockr.yml
   ├── .gitignore
   ├── .travis.yml
   ├── README.md
   ├── ballet.yml
   ├── setup.py
   ├── src
   │   └── myproject
   │       ├── __init__.py
   │       ├── __main__.py
   │       ├── api.py
   │       ├── features
   │       │   ├── __init__.py
   │       │   ├── contrib
   │       │   │   └── __init__.py
   │       │   └── encoder.py
   │       └── load_data.py
   └── tasks.py

   5 directories, 15 files

Importantly, by keeping this project structure intact, Ballet will be able to automatically care
for your feature engineering pipeline.

* ``ballet.yml``: a Ballet configuration file, with details about the prediction problem, the
  training data, and location of feature engineering source code.
* ``.travis.yml``: a `Travis CI`_ configuration file pre-configured to run a Ballet validation
  suite.
* ``src/myproject/api.py``: this is where Ballet will look for functionality implemented by your
  project, including a function to load training/test data or collected features. Stubs for this
  functionality are already provided by the template but you can further adapt them.

Project installation
~~~~~~~~~~~~~~~~~~~~

For local development, you can then install your project. This will make your feature
engineering pipeline accessible in interactive settings (Python interpreter, Jupyter notebook)
and as a command-line tool.

.. code-block:: console

   $ cd ballet-my-project
   $ conda create -n myproject -y && conda activate myproject  # or your preferred environment tool
   (myproject) $ pip install invoke && invoke install

Collaboration via git and GitHub
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Under the hood, contributors will collaborate using the powerful functionality provided by git
and GitHub. In fact, after the quickstart step, you already have a git-tracked repository and a
git remote set up.

.. code-block:: console

   $ git log
   commit 5c8ec6773aff4030fc1256a7c9e13675d620bb6e (HEAD -> master, project-template)
   Author: Jane Developer <jane@developer.org>
   Date:   Tue Apr 16 17:27:44 2019 -0400

       Automatically generated files from ballet quickstart

   $ git remote -v
   origin	git@github.com:jane_developer/ballet-my-project (fetch)
   origin	git@github.com:jane_developer/ballet-my-project (push)

Next, you must create the matching GitHub project, ``myproject``, under the account of the
``github_owner`` that you specified earlier (in this case, ``jane_developer``). Do not
initialize the project with any sample files that GitHub offers.

After you having created the project on GitHub, push your local copy.

.. code-block:: console

   $ git push --all origin


Enabling continuous integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ballet makes uses of the continuous integration service `Travis CI`_ in order to validate code
that contributors propose as well as perform streaming feature definition selection. You must
enable Travis CI for your project on GitHub by`following these simple directions
<https://docs.travis-ci.com/user/tutorial/#to-get-started-with-travis-ci-using-github>`_. You can
skip any steps that have to do with customizing the ``.travis.yml`` file, as we have already done
that for you in the quickstart.

Installing bots
~~~~~~~~~~~~~~~

Many Ballet project use bots to assist maintainers.

1. Ballet bot. Install it `here <https://github.com/apps/ballet-bot>`__. Ballet bot will
automatically merge or close PRs based on the CI test result and the project settings configured
in the ``ballet.yml`` file.

2. Repolockr. Install it `here <https://github.com/apps/repolockr>`__. Repolockr checks every PR
to ensure that "protected" files have not been changed. These are files listed in the Repolockr
config file on the master branch. A contributor might accidentally modify a protected file like
``ballet.yml`` which could break the project or the CI pipeline; Repolockr will detect this and
fail the PR which might accidentally pass otherwise.

Developing new features
-----------------------

At this point, your feature engineering pipeline contains no features. How will your
contributors add more?

Using any of a number of development workflows, contributors write new features and submit them
to your project for validation. For more details on the contributor workflow, see :doc:`/contributor_guide`.

Validating features
~~~~~~~~~~~~~~~~~~~

The ``ballet-my-project`` repository has received a new pull request which triggers an automatic
evaluation.

1. The PR is examined by the CI service.
2. The ``ballet validate`` command is run, which validates the proposed feature contribution using
   functionality within the ``ballet.validation`` package.
3. If the feature can be validated successfully, the PR passes, and the proposed feature can be
   merged into the project.

Pruning features
~~~~~~~~~~~~~~~~

Once a feature has been accepted and merged into your project's master branch, it may mean that
an older feature has now become "redundant": the new feature is providing all of the information
contained in the old feature, and more.

1. Each commit to master is examined by the CI service.
2. The ``ballet validate`` command is run and automatically determines whether the commit is a
   merge commit that comes from merging an accepted feature.
3. If so, then the set of existing features is pruned to remove redundant features.
4. Pruned features are automatically deleted from your source repository by an automated service.

Applying the feature engineering pipeline
-----------------------------------------

As your repository fills with features, your feature engineering pipeline is always available to
engineer features from new data points or datasets.

For interactive usage:

.. code-block:: python

   from myproject.api import build, load_data

   # load training data
   X_df_tr, y_df_tr = load_data()

   # fit pipeline to training data
   result = build(X_df_tr, y_df_tr)
   pipeline, encoder = result.pipeline, result.encoder

   # load new data and apply pipeline
   X_df, y_df = load_data(input_dir='/path/to/new/data')
   X = pipeline.transform(X_df)
   y = encoder.transform(y_df)

For command-line usage:

.. code-block:: console

   $ python -m myproject engineer-features path/to/test/data path/to/features/output

Updating the framework
----------------------

If there are updates to the Ballet framework after you have started working on your project, you
can access them easily.

First, update the ``ballet`` package itself using the usual ``pip`` mechanism:

.. code-block:: console

   $ pip install --upgrade ballet

Pip will complain that the upgraded version of ballet is incompatible with the version required
by the installed project. That is okay, as we will presently update the project itself to work
with the new version of ballet.

Next, use the updated version of ``ballet`` to incorporate any updates to the "upstream" project
template used to create new projects.

.. code-block:: console

   $ ballet update-project-template --push

This command will re-render the project template using the saved inputs you have provided in the
past and then safely merge it first to your ``project-template`` branch and then to your
``master`` branch. Finally, given the ``--push`` flag it will push updates to
``origin/master`` and ``origin/project-template``. The usage of this command is described in more
detail `here <cli_reference.html#ballet-update-project-template>`__.


.. _cookiecutter: https://cookiecutter.readthedocs.io/en/latest
.. _`Travis CI`: https://travis-ci.com
