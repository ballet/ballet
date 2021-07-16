===
FAQ
===

These questions are directed towards contributors to a feature engineering project.

General
-------

How can I check the performance of a feature?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ballet projects have two built-in routines to validate features. You can access them using the Ballet interactive client :py:data:`ballet.b`.

1. :py:meth:`~ballet.client.Client.validate_feature_api`
2. :py:meth:`~ballet.client.Client.validate_feature_acceptance`

You can use these methods in your development environment. You can also use any other methods you think are useful, such as extracting feature values from training data and making a scatter plot, computing :math:`R^2` and mutual information, etc.

See also: :ref:`contributor_guide:Test your feature (local)`.

There was an error submitting my feature using Assemblé in Jupyter Lab.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here are common situations:

1. Problem: You selected a cell in your notebook that did not contain the feature. It may have contained descriptive text or other Python code.

   .. image:: _static/assemble_error_not_valid_python_code.png

   Solution: Select the code cell that contains the feature and try again. Make sure the feature you want to previewed in the "Submit Feature?" dialog.

2. Problem: The submission fails with some other cryptic message that appears to be a Python exception.

   Solution: Please report this error -- thanks for your help!

   1. `Open a new issue on the ballet-assemble project <https://github.com/ballet/ballet-assemble/issues/new>`__

   2. Open the web console. (See `Open the web console on Firefox <https://developer.mozilla.org/en-US/docs/Tools/Web_Console#opening_the_web_console>`__ or `Open the web console on Chrome <https://developer.chrome.com/docs/devtools/open/#console>`__.)

   3. Scroll to the bottom of the console log. Copy the detailed error information. It should be an object that has as its ``message`` field the same message you just saw in the popup.

   4. Paste the detailed error message in the indicated location in the new issue. Then link to the Ballet project on GitHub you are working on, and provide any more description that could be helpful.

My feature has a valid API locally, why was it rejected?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In your local environment, your feature is validated in the context of the other code you have written so far. But in CI, it is tested in an isolated environment. So usually, if your feature is rejected in CI, it means that it was relying on other code that was present in your environment but was not part of your submission. For example, modules that you imported or helper functions that you used.

In this example, the user relies on ``np`` but it does not exist in their feature snippet. The solution is to move the ``import numpy as np`` into their submission (i.e. their code cell in a notebook).

.. code-block:: python

   from ballet import Feature
   # import numpy as np   # uncomment this line to fix!

   input = ['A', 'B']
   transformer = lambda df: df.apply(np.sum, axis=1)  # np is not available here!
   feature = Feature(input=input, transformer=transformer)

See also: :ref:`contributor_guide:Understanding Validation Results`

Why did my feature fail the feature acceptance validation?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The feature acceptance validation checks to see if the feature values that your feature extracts contribute to the machine learning goals. If the feature fails this validation, it means that perhaps the extracted feature values are not predictive of the target, or that they are redundant with another feature, or that the benefit from including the feature in the model is outweighed by the increase in dimensionality of the columns.

See also: :ref:`contributor_guide:Understanding Validation Results`

Are there any differences between local validation and CI validation?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some Ballet projects have configured their validation such that in CI a separate, held-out dataset is used. You can see if this is the case by looking for the existence of the ``validation.split`` key in the project's ``ballet.yml`` configuration file. If your feature passes all validation locally but fails in CI, your feature might be "overfitting" the development set, either in an ML sense or by not considering a "change" in the data distribution that might lead to missing values appearing where you had not expected or similar.

My feature relies on a new library, how can I add it to the project?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, confirm that the new library is not already a dependency of the project by running ``pip freeze`` from within the virtual environment and confirming your desired library is absent.

If your feature must use this new library, first install it locally and ensure that the feature you develop using that dependency is satisfactory.

Then, before submitting the feature to the project, submit a separate PR that adds the dependency to the project's ``setup.py`` file, as illustrated by this diff:

.. code-block:: diff

   --- a/setup.py
   +++ b/setup.py
   @@ -2,6 +2,7 @@ from setuptools import setup, find_packages

    requirements = [
        'ballet[all]==0.7.9',
   +    'newlibrary>=4.7',
    ]

A maintainer will manually review the PR and must merge it before you can then submit your feature. (Otherwise your feature will fail due to the missing dependency.)

How do I delete an already-accepted feature?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are a few reasons you might want to delete a feature that has already been accepted:

1. You submitted a duplicate feature, and the validation that was configured for your project considers each feature in isolation so it was accepted.
2. You realized there was an error with your feature even though it passed validation.
3. You have an idea to improve this feature and want to delete it and start over.

To delete a feature, just introduce a pull request that deletes the file containing the feature definition, either using the GitHub UI or the git client of your choice. Validation may fail (because it usually expects that you are proposing to add a new feature, rather than make other changes), but a maintainer will manually review your proposal.

How do I edit a feature definition that has been rejected?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are using Assemblé, there is unfortunately no support for this currently. Here are two workaround:

1. Copy the feature definition from this PR into a new Assemble session, make fixes, submit a new PR, and close this one (easy)
2. Push additional commits to this PR using the local feature development workflow (advanced)
    1. clone your fork (``git clone https://github.com/<your user name>/<ballet project name>.git``)
    2. checkout this branch (``git checkout -t origin/submit-feature-<the id of the feature branch>``)
    3. edit the file and commit changes
    4. push commits back to the branch on your fork (``git push``)
    5. the PR will be automatically updated by your new commits and validation will run again

Developing features
-------------------

How can I learn to write better features?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The way that feature engineering code is written in Ballet may be unfamiliar at first.

Make sure to review the :doc:`feature_engineering_guide`.

If you are coming from a background of using *pandas* for feature engineering, make sure to look over the :ref:`feature_engineering_guide:Differences from Pandas`.

Aim to *learn by example* by reading existing feature definitions written by your collaborators.

How do I debug a failing ``CanTransformOneRowCheck``?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, make sure you can replicate the failing check by trying to use your feature to transform one row of data and observe the failure:

.. code-block:: python

   pipeline = feature.pipeline
   row = X_df.iloc[0:1, :]
   pipeline.fit(X_df, y_df)
   pipeline.transform(row)

Perhaps the traceback will help you realize your error immediately.

If not, consider places where you have made assumptions about the shapes of different objects passing through your transformer steps. In the "one row" case, the input to your transformer is a data frame that has shape ``(1, m)``.

* Are there places where your code will implicitly reshape this as a series or 1-d array rather than a data frame or 2-d array?
* Have you assumed that each column will contain some non-null values, but now that you receive a single row as input, any null values will cause your feature to fail? If so, make sure you are learning how to impute missing data on the training set and storing any parameters.

Consider the difference in this example:

.. code-block:: python

   def transformer(df):
       # bad - at inference time, df may be a single row with nulls, and the
       # mean is also null
       return df.fillna(df.mean())

   # better - you are learning the mean from the training data rather than the
   # test data
   from ballet.eng.external import SimpleImputer
   transformer = SimpleImputer(strategy='mean')
