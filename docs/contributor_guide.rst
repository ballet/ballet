=================
Contributor Guide
=================

If you are on this page, you are probably thinking about contributing to a Ballet feature
engineering collaboration. Welcome!

Here are the steps to develop a new feature for inclusion in a Ballet project. In this guide, we
will use the real example of `Predict House Prices`_, a feature engineering collaboration to be
used for predicting the sale price of houses in Ames, Iowa, given raw data about each house.

.. tip::

   Please ask for help on our Gitter chat!

   .. image:: https://badges.gitter.im/ballet-project/community.svg
      :alt: Chat on Gitter
      :target: https://gitter.im/ballet-project/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge


Cloud Feature Development Workflow
==================================

Some Ballet projects may have set up a development workflow using
`Assemblé`_, a development environment for Ballet on top of Jupyter Lab,
with a hosted Jupyter Lab environment running on Binder and an in-Lab
submission extension. This greatly simplifies the development workflow at
the cost of some flexibility.

Launch Assemblé
---------------

Click the |launch-assemble| link in the project's README to launch an interactive notebook. This may take a few minutes while Binder builds the repository -- grab a cup of coffee!

Work in a notebook
------------------

Begin editing the demo notebook or create a new notebook.

Authenticate with GitHub
------------------------

Authenticate with GitHub by clicking the GitHub icon on the right side of
the Notebook toolbar as shown. This will initiate an authentication process
in which you will be able to log into your GitHub account and authorize
``ballet-github-oauth-gateway`` for limited access to your account. As a
result of this secure process, the Ballet Lab extension will be able to
submit your feature on your behalf to the upstream project.

.. image:: _static/auth_with_github.gif
   :alt: An animation of a user proceeding with the GitHub authentication flow in a demo notebook.
   :align: center
   :scale: 75%

When you have successfully authenticated, the GitHub icon will change color
to green.

.. _`contributor_guide:Develop a new feature (cloud)`:

Develop a new feature
---------------------

Develop a new feature in the notebook. See :ref:`contributor_guide:Write your feature` below, and the project may have prompts in the notebook to guide you. See also the :doc:`feature_engineering_guide`.

.. _`contributor_guide:Test your feature (cloud)`:

Test your feature
-----------------

Test your feature. See :ref:`contributor_guide:Test your feature (local)`
below, and the project may have prompts in the notebook to guide you.

Submit your feature
-------------------

Submit your feature.

First, *select the code cell* that contains the feature.

Next, locate the submission button provided by the Jupyter Lab extension, as
shown in the screenshot.

.. image:: _static/assemble_submit_button_annotated_submit.png
   :alt: The submission button illustrated in a notebook.
   :align: center
   :scale: 75%

When you are ready, press the "Submit" button. You will be asked to confirm
that you want to submit the feature.

If the submission succeeds, you will get a link to the pull request that is
associated with your feature. If it fails, you will get an explanation of
the failure.

.. note::

   The content of the cell must be a standalone Python module, as it will be placed in an
   empty Python source file. This means that any imports or helper functions must be defined
   (or re-defined) within this cell, otherwise your submitted feature will fail to validate
   due to missing imports/helpers.

Local Feature Development Workflow
==================================

The most flexible and powerful development workflow is based on developing
locally making heavy use of your command line. There are two parts to the
workflow. First, you setup your development environment (only do this once).
Second, you develop a new feature (repeat the steps every time you create a
new feature).

This section discusses the use of many concepts commonly used in software
development, including git, GitHub, make, and virtual environments.

Setup your development environment
----------------------------------

Fork the project
^^^^^^^^^^^^^^^^

For example, use the GitHub UI, or the ``hub`` cli.

Clone your fork
^^^^^^^^^^^^^^^

Clone your fork in your local development environment:

.. code-block:: console

   $ git clone https://github.com/your_user_name/ballet-predict-house-prices
   $ cd ballet-predict-house-prices
   $ git remote add upstream https://github.com/HDI-Project/ballet-predict-house-prices

Create a virtualenv
^^^^^^^^^^^^^^^^^^^

Create and activate a new virtual environment using your environment manager of choice, such as `conda`_.

.. code-block:: console

   $ conda create -y -n myenv python=3.7
   $ conda activate myenv

.. _`contributor_guide:Develop a new feature (local)`:

Develop a new feature
---------------------

Update and install the project
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This will make the feature engineering pipeline accessible in interactive
settings (Python interpreter, Jupyter notebook) and as a command-line tool.

.. code-block:: console

   (myenv) $ git checkout master
   (myenv) $ git pull upstream master
   (myenv) $ git push origin master
   (myenv) $ pip install invoke && invoke install

.. note::

   You should repeat the entirety of this step every time before you begin
   working on a new feature, in order to synchronize changes made to the
   upstream project, such as the introduction of new features by other
   collaborators or an update to the Ballet framework itself.

Start working on a new feature
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

   (myenv) $ git checkout -b develop-my-feature
   (myenv) $ ballet start-new-feature
   Starting new feature...
   username [your_user_name]:
   featurename [featurename]: impute_lot_frontage
   2019-12-11 10:56:00,517 INFO - Start new feature successful.
   2019-12-11 10:56:00,517 INFO - Created src/ballet_predict_house_prices/features/contrib/user_your_user_name/feature_impute_lot_frontage.py
   Starting new feature...DONE

This will create a new Python module within the project's "contrib"
directory to hold your feature.

* The contrib directory is named like ``src/<ballet_project>/features/contrib``.
* The new subpackage must be named like ``user_<github username>``.
* The new submodule that will contain the feature must be named like ``feature_<feature name>.py``.

Write your feature
^^^^^^^^^^^^^^^^^^

We call the code you write to extract one group of related feature values a
*feature definition*, or simply *feature*. Within your feature submodule, you can write arbitrary
Python code. Ultimately, a single object that is an instance of
``ballet.Feature`` must be defined; it will be imported by the feature
engineering pipeline.

In this example, a feature is defined that receives column ``'Lot
Frontage'`` from the data and imputes missing values with the mean of the
training data.

.. include:: fragments/feature-engineering-guide-second-feature.py
   :code: python

.. tip::

   For a full tutorial on feature engineering in Ballet, check out the separate :doc:`feature_engineering_guide`.

Only the Python packages that are existing dependencies of the project can be used in feature engineering. Otherwise, if the feature were to be accepted, then the feature engineering pipeline would break due to a missing dependency. Usually, the dependencies of a Ballet project are the core ``ballet`` package and its own dependencies. If Ballet is installed with the ``[all]`` extra, then it additional re-exports feature engineering primitives from many common libraries. See :py:mod:`ballet.eng.external` for a summary.

If you must add a new dependency, see :ref:`faq:My feature relies on a new library, how can I add it to the project?`.

.. _`contributor_guide:Test your feature (local)`:

Test your feature
^^^^^^^^^^^^^^^^^

Observe later in this guide that when you submit your feature, there will be
four separate validation steps. In your local development environment, you
can check two of them: whether the feature you have written satisfies the
"feature API", and whether the feature contributes positively to the ML
performance of the feature engineering pipeline.

To validate your feature, Ballet provides a client ``b`` for easy access to
validation methods. It takes as input the feature and runs a series of tests
to make sure that the feature works correctly. You can optionally pass
specific entities and labels to use as well.

.. code-block:: python

   from ballet import b
   b.validate_feature_api(feature)
   # True


Second, the function ``validate_feature_acceptance`` takes as input the
feature object and runs an algorithm to determine whether the existing
feature engineering pipeline for the Ballet project that you are working
on performs better with or without your feature.

.. code-block:: python

   from ballet import b
   b.validate_feature_acceptance(feature)
   # True

Under the hood, it tries to automatically detect the Ballet project that you
are working on and builds the existing feature engineering pipeline that is
part of the project. It also loads the specific feature accepter that has
been configured for your project.

To gain additional insight into any of the validation procedures, including
details on why your feature may have failed to validate, enable ballet
logging.

.. code-block:: python

   from ballet.util.log import enable
   enable(level='INFO')   # or, level='DEBUG'
   # [2019-12-22 10:51:30,336] {ballet: log.py:34} INFO - Logging enabled at level INFO.

Submit the feature
------------------

To submit your feature, you have two options.

Option 1: Git workflow
^^^^^^^^^^^^^^^^^^^^^^

In this workflow, you work with git directly to commit and push your change and open a pull request with the upstream project repo.

Commit your changes
"""""""""""""""""""

.. code-block:: console

   (myenv) $ git add .
   (myenv) $ git commit -m "Add my new feature"
   (myenv) $ git push origin develop-my-feature

Create a pull request
"""""""""""""""""""""

Create a pull request to the project repository.

The output of the ``git push`` command will include a link to open a new
pull request on the upstream project. Navigate to the url in your browser
and open a new PR. Alternately, you can use the command-line tool `hub`_:

.. code-block:: console

   (myenv) $ hub pull-request

Option 2: In-Lab Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^

In this workflow, you use Ballet Assemblé in order to submit
code directly from within your analysis notebook if you are developing in
Jupyter Lab. This has the same user experience as described in the cloud
feature development workflow above. Here, you should `install
ballet-assemble`_ following the directions in that project.
Importantly, you must authorize the extension to interact with GitHub on
your behalf in one of three ways:

#. use the built-in GitHub OAuth functionality to obtain a new OAuth token
   with one click

#. obtain a GitHub OAuth token yourself and populate the variable
   ``$GITHUB_TOKEN``

#. obtain a GitHub OAuth token and pass it as an option when starting
   JupyterLab using ``--AssembleApp.github_token=$TOKEN``.

Understanding Validation Results
================================

Once you have developed and submitted a feature, Ballet will validate it in four steps in an isolated continuous integration environment.

#. Check feature API: does your feature behave properly on expected and unexpected inputs? For example, it should not produce feature values with NaNs or throw errors on well-formed data instances.

#. Check project structure: does your PR respect the project structure, that is, you have created valid Python modules at the right path, etc.

#. Evaluate feature acceptance: do the feature values that your feature extracts contribute to the machine learning goals? Depending on the configuration of the upstream project, the project may evaluate your features in a more or less aggressive manner, ranging from accepting all features to accepting only those that produce an information gain greater than some threshold.

#. Evaluate feature pruning: does the introduction of your feature cause other features to be unnecessary? If so they may be pruned.

Depending on the configuration of the upstream project, you will see various "bots" act on
these steps. If your PR passes the first three steps, the `Ballet Bot`_ may approve and merge
your PR automatically. If your PR is merged, the Ballet Bot may automatically prune features
from the master branch. If your feature is rejected, you can inspect the logs produced by the
Travis CI service to see what went wrong. (We are working on improving the user experience of
this debugging.)

Conclusion
==========

In this guide, we walked through all of the steps required to submit your first feature to a
ballet collaboration.

.. figure:: https://upload.wikimedia.org/wikipedia/en/f/f8/Internet_dog.jpg
   :width: 300
   :align: center
   :alt: "On the internet, nobody knows you're a dog" cartoon

   Image from *The New Yorker* cartoon by Peter Steiner, 1993, via Wikipedia.

.. _`Assemblé`: https://github.com/HDI-Project/ballet-assemble
.. |launch-assemble| image:: _static/launch-assemble.svg
.. _`Predict House Prices`: https://github.com/HDI-Project/ballet-predict-house-prices
.. _`conda`: https://conda.io/en/latest/
.. _`hub`: https://hub.github.com/
.. _`Ballet Bot`: https://github.com/apps/ballet-bot
.. _`install ballet-assemble`: https://github.com/HDI-Project/ballet-assemble/blob/master/README.md
