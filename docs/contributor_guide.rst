=================
Contributor Guide
=================

If you are on this page, you are probably thinking about contributing to a Ballet feature
engineering collaboration. Welcome!

Here are the steps to develop a new feature for inclusion in a Ballet project. In this guide, we
will use the real example of `Predict House Prices`_, a feature engineering collaboration to be
used for predicting the sale price of houses in Ames, Iowa, given raw data about each house.

.. tip::

   This guide discusses the use of many concepts commonly used in software development, including
   git, GitHub, make, and virtual environments. We are working on supporting a higher-level
   workflow in which you are able to develop and collaborate entirely from within your Jupyter
   notebook/lab environment. Please feel free to ask for help on our Slack channel!

   .. image:: https://img.shields.io/static/v1?label=chat&message=on%20slack&color=4A154B&logo=slack
      :alt: Chat on Slack
      :target: https://slack.com/share/IUQBPT316/gXn5PKAJGNnqfX0Mz1oKDRDJ/enQtOTc2Mzk3OTIxMDQwLTUxNDUzNmUxMzY0YTJmNGFiMGFmNGI3YWIyOWY2ZDZjYzRhOGE4MGVjYzA4ZDQ4ZjRkNDE0OTQ2ZTRmMzJmNjA


Cloud Feature Development Workflow
==================================

Some Ballet projects may have set up a development workflow using a hosted Jupyter Lab
environment and an in-lab submission extension. This greatly simplifies the development workflow
at the cost of some flexibility.

#. Click the |launch-binder| link in the project's README to launch an interactive notebook.

#. Open a demo notebook or create a new notebook.

#. Develop a new feature. See step below, "Write your feature".

#. Test your feature. See step below, "Test your feature".

#. Submit your feature. Find the submission button provided by the Jupyter Lab extension, as
   shown in the screenshot.

   .. image:: _static/labextension_submit_button_annotated.png
      :alt: The submission button illustrated in a notebook.
      :align: center
      :scale: 75%

   When you have finished developing your feature, select the code cell that contains the feature
   and press the "Submit" button. You will be asked to confirm that you want to submit the feature.

   .. note::

      The content of the cell must be a standalone Python module, as it will be placed in an
      empty Python source file. This means that any imports or helper functions must be defined
      (or re-defined) within this cell, otherwise your submitted feature will fail to validate
      due to missing imports/helpers.

.. |launch-binder| image:: https://mybinder.org/badge_logo.svg

Local Feature Development Workflow
==================================

The most flexible and powerful development workflow is based on developing locally making heavy
use of your command line. There are two parts to the workflow. First, you setup your development
environment (only do this once). Second, you develop a new feature (repeat the steps every time you
create a new feature).


Setup your development environment
-----------------------------------

#. Fork the project.

#. Clone your fork in your local development environment.

   .. code-block:: console

      $ git clone https://github.com/your_user_name/ballet-predict-house-prices
      $ cd ballet-predict-house-prices
      $ git remote add upstream https://github.com/HDI-Project/ballet-predict-house-prices

#. Create and activate a new virtual environment using your environment manager of choice, such
   as `conda`_. **(Strongly recommended.)**

   .. code-block:: console

      $ conda create -n myenv python=3.7 -y && conda activate myenv


Develop a new feature
---------------------

#. Update and install the project. This will make the feature engineering pipeline accessible in
   interactive settings (Python interpreter, Jupyter notebook) and as a command-line tool.

   .. code-block:: console

      (myenv) $ git checkout master
      (myenv) $ git pull upstream master
      (myenv) $ git push origin master
      (myenv) $ make install

   .. note::

      You should repeat the entirety of this step every time before you begin working on a new
      feature, in order to synchronize changes made to the upstream project, such as the
      introduction of new features by other collaborators or an update to the ballet framework
      itself.

#. Start working on a new feature.

   .. code-block:: console

      (myenv) $ git checkout -b develop-my-feature
      (myenv) $ ballet start-new-feature
      Starting new feature...
      username [your_user_name]:
      featurename [featurename]: impute_lot_frontage
      2019-12-11 10:56:00,517 INFO - Start new feature successful.
      2019-12-11 10:56:00,517 INFO - Created src/ballet_predict_house_prices/features/contrib/user_your_user_name/feature_impute_lot_frontage.py
      Starting new feature...DONE

   This will create a new Python module within the project's "contrib" directory to hold your
   feature.

   * The contrib directory is named like ``src/<ballet_project>/features/contrib``.
   * The new subpackage must be named like ``user_<github username>``.
   * The new submodule that will contain the feature must be named like ``feature_<feature name>.py``.

#. Write your feature. We call the code you write to extract one group of related feature values
   a *logical feature*. Within your feature submodule, you can write arbitrary Python code.
   Ultimately, a single object that is an instance of ``ballet.Feature`` must be defined; it will
   be imported by the feature engineering pipeline.

   In this example, a feature is defined that receives column ``'Lot Frontage'`` from the
   data and imputes missing values with the mean of the training data.

   .. code-block:: python

      from ballet import Feature
      from sklearn.impute import SimpleImputer

      input = ["Lot Frontage"]
      transformer = SimpleImputer(strategy="mean")
      name = "Imputed Lot Frontage"
      feature = Feature(input=input, transformer=transformer, name=name)

   .. tip::

      For a full tutorial on feature engineering in Ballet, check out the separate
      :doc:`Feature Engineering Guide <./feature_engineering_guide>`.

#. Test your feature. Observe later in this guide that when you submit your feature, there will be
   four separate validation steps. In your local development environment, you can check two of
   them: whether the feature you have written satisfies the "feature API", and whether the
   feature contributes positively to the ML performance of the feature engineering pipeline.

   First, the function ``validate_feature_api`` takes as input the feature object and some training
   data and runs a series of tests to make sure that the feature works correctly.

   .. code-block:: python

      from ballet.validation.feature_api import validate_feature_api
      validate_feature_api(feature, X_df, y_df)
      # True


   Second, the function ``validate_feature_acceptance`` takes as input the feature object and
   some training data, and runs an algorithm to determine whether the existing feature
   engineering pipeline for the Ballet project that you are working on performs better with or
   without your feature.

   .. code-block:: python

      from ballet.validation.feature_acceptance import validate_feature_acceptance
      validate_feature_acceptance(feature, X_df, y_df)
      # True

   Under the hood, it tries to automatically detect the Ballet project that you are working
   on and builds the existing feature engineering pipeline that is part of the project. It also
   loads the specific feature accepter that has been configured for your project.

   To gain additional insight into any of the validation procedures, including details on
   why your feature may have failed to validate, enable ballet logging.

   .. code-block:: python

      from ballet.util.log import enable
      enable(level='INFO')   # or, level='DEBUG'
      # [2019-12-22 10:51:30,336] {ballet: log.py:34} INFO - Logging enabled at level INFO.

#. Submit your feature. Commit your changes and create a pull request to the project repository.

   .. code-block:: console

      (myenv) $ git add .
      (myenv) $ git commit -m "Add my new feature"
      (myenv) $ git push origin develop-my-feature

   The output of the ``git push`` command will include a link to open a new pull request on the
   upstream project. Navigate to the url in your browser and open a new PR. Alternately, you can
   use the command-line tool `hub`_:

   .. code-block:: console

      (myenv) $ hub pull-request

#. Observe the validation results. Ballet will now validate your feature in four steps.

   1. Check feature API: does your feature behave properly on expected and unexpected inputs?
      For example, it should not produce feature values with NaNs or throw errors on well-formed
      data instances.

   2. Check project structure: does your PR respect the project structure, that is, you have
      created valid Python modules at the right path, etc.

   3. Evaluate feature acceptance: do the feature values that your feature extracts contribute
      to the machine learning goals? Depending on the configuration of the upstream project, the
      project may evaluate your features in a more or less aggressive manner, ranging from
      accepting all features to accepting only those that produce an information gain greater
      than some threshold.

   4. Evaluate feature pruning: does the introduction of your feature cause other features to be
      unnecessary? If so they may be pruned.

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

.. _`Predict House Prices`: https://github.com/HDI-Project/ballet-predict-house-prices
.. _`conda`: https://conda.io/en/latest/
.. _`hub`: https://hub.github.com/
.. _`Ballet Bot`: https://github.com/apps/ballet-bot
