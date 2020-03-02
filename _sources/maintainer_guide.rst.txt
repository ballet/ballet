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

Then, `install Ballet <Installation.html>`__ on your development machine.

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

   $ tree -a ballet-my-project/ -I .git
   ballet-my-project/
   ├── .cookiecutter_context.json
   ├── .github
   │   └── repolockr.yml
   ├── .gitignore
   ├── .travis.yml
   ├── Makefile
   ├── README.md
   ├── ballet.yml
   ├── setup.py
   └── src
       └── myproject
           ├── __init__.py
           ├── features
           │   ├── __init__.py
           │   └── contrib
           │       └── __init__.py
           └── load_data.py

   5 directories, 12 files

Importantly, by keeping this project structure intact, Ballet will be able to automatically care
for your feature engineering pipeline.

* ``ballet.yml``: a Ballet configuration file, with details about the prediction problem, the
  training data, and location of feature engineering source code.
* ``.travis.yml``: a `Travis CI`_ configuration file pre-configured to run a Ballet validation
  suite.
* ``src/myproject/load_data.py``: this is where you will write code to load training data
* ``src/myproject/features/contrib``: this is where the features created by your project's
  contributors will live.

Project installation
~~~~~~~~~~~~~~~~~~~~

For local development, you can then install your project. This will make your feature
engineering pipeline accessible in interactive settings (Python interpreter, Jupyter notebook)
and as a command-line tool.

.. code-block:: console

   $ cd myproject
   $ conda create -n myproject -y && conda activate myproject  # or your preferred environment tool
   (myproject) $ make install

Collaboration via git and GitHub
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Under the hood, contributors will collaborate using the powerful functionality provided by git
and GitHub. In fact, after the quickstart step, you already have a git-tracked repository and a
git remote set up.

.. code-block:: console

   $ cd myproject

   $ git log
   commit 5c8ec6773aff4030fc1256a7c9e13675d620bb6e (HEAD -> master, project-template)
   Author: Jane Developer <jane@developer.org>
   Date:   Tue Apr 16 17:27:44 2019 -0400

       Automatically generated files from ballet-quickstart

   $ git remote -v
   origin	git@github.com:jane_developer/myproject (fetch)
   origin	git@github.com:jane_developer/myproject (push)

Next, you must create the matching GitHub project, ``myproject``, under the account of the
``github_owner`` that you specified earlier (in this case, ``jane_developer``). Do not
initialize the project with any sample files that GitHub offers.

After you having created the project on GitHub, push your local copy.

.. code-block:: console

   $ git push origin master


Enabling continuous integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ballet makes uses of the continuous integration service `Travis CI`_ in order to validate code
that contributors propose as well as perform streaming logical feature selection. You must
enable Travis CI for your project on GitHub by `following these simple directions <https://docs
.travis-ci.com/user/tutorial/#to-get-started-with-travis-ci>`_. You can skip any steps that have
to do with customizing the ``.travis.yml`` file, as we have already done that for you in the
quickstart.


Developing new features
-----------------------

At this point, your feature engineering pipeline contains no features. How will your
contributors add more?

Using any of a number of development workflows, contributors write new features and submit them
to your project for validation. For more details on the contributor workflow, see `Contributor
Guide`_.

Validating features
~~~~~~~~~~~~~~~~~~~

The ``myproject`` repository has received a new pull request which triggers an automatic
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

   import myproject

   # load training data and fit pipeline
   X_df_tr, y_df_tr = myproject.load_data.load_data()
   out = myproject.features.build(X_df_tr, y_df_tr)
   mapper_X = out.mapper_X
   encoder_y = out.encoder_y

   # load new data and apply pipeline
   X_df, y_df = myproject.load_data.load_data(input_dir='/path/to/new/data')
   X = mapper_X.transform(X_df)
   y = encoder_y.transform(y_df)

For command-line usage:

.. code-block:: console

   $ myproject-engineer-features path/to/test/data path/to/features/output

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
detail `here <cli_reference .html#ballet-update-project-template>`_.


.. _cookiecutter: https://cookiecutter.readthedocs.io/en/latest
.. _`Travis CI`: https://travis-ci.com
.. _`pull request`: https://help.github.com/articles/about-pull-requests/
.. _`Contributor Guide`: contributor_guide.html
