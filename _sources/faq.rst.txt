===
FAQ
===

These questions are directed towards contributors to a feature engineering project.

How can I check the performance of a feature?
---------------------------------------------

Ballet projects have two built-in routines to validate features. You can access them using the Ballet interactive client :py:data:`ballet.b`.

1. :py:meth:`~ballet.client.Client.validate_feature_api`
2. :py:meth:`~ballet.client.Client.validate_feature_acceptance`

You can use these methods in your development environment. You can also use any other methods you think are useful, such as extracting feature values from training data and making a scatter plot, computing R^2 and mutual information, etc.

See also: :ref:`contributor_guide:Test your feature (local)`.

There was an error submitting my feature using Assemblé in Jupyter Lab.
-----------------------------------------------------------------------

Here are common situations:

1. Problem: You selected a cell in your notebook that did not contain the feature. It may have contained descriptive text or other Python code.

   .. image:: _static/assemble_error_not_valid_python_code.png

   Solution: Select the code cell that contains the feature and try again. Make sure the feature you want to previewed in the "Submit Feature?" dialog.

2. Problem: The submission fails with some other cryptic message that appears to be a Python exception.

   Solution: Please report this error -- thanks for your help!

   1. `Open a new issue on the ballet-assemble project <https://github.com/HDI-Project/ballet-assemble/issues/new>`__

   2. Open the web console. (See `Open the web console on Firefox <https://developer.mozilla.org/en-US/docs/Tools/Web_Console#opening_the_web_console>`__ or `Open the web console on Chrome <https://developers.google.com/web/tools/chrome-devtools/open#console>`__.)

   3. Scroll to the bottom of the console log. Copy the detailed error information. It should be an object that has as its ``message`` field the same message you just saw in the popup.

   4. Paste the detailed error message in the indicated location in the new issue. Then link to the Ballet project on GitHub you are working on, and provide any more description that could be helpful.

My feature has a valid API locally, why was it rejected?
--------------------------------------------------------

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
----------------------------------------------------------

The feature acceptance validation checks to see if the feature values that your feature extracts contribute to the machine learning goals. If the feature fails this validation, it means that perhaps the extracted feature values are not predictive of the target, or that they are redundant with another feature, or that the benefit from including the feature in the model is outweighed by the increase in dimensionality of the columns.

See also: :ref:`contributor_guide:Understanding Validation Results`

My feature relies on a new library, how can I add it to the project?
--------------------------------------------------------------------

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

How can I learn to write better features?
-----------------------------------------

The way that feature engineering code is written in Ballet may be unfamiliar at first.

Make sure to review the :doc:`feature_engineering_guide`.

If you are coming from a background of using *pandas* for feature engineering, make sure to look over the :ref:`feature_engineering_guide:Pandas ⇔ Ballet Examples`.
