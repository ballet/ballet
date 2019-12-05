=================
Contributor Guide
=================

If you are on this page, you are probably thinking about contributing to a Ballet
feature engineering collaboration. Welcome!

Here are the steps to develop a new feature for inclusion in a Ballet project.

#. Fork the project and clone your fork in your local development environment.
#. Install the project in a virtual environment. This will make the feature
    engineering pipeline accessible in interactive settings (Python interpreter, Jupyter notebook)
    and as a command-line tool.

    .. code-block:: console

       $ cd myproject
       $ conda create -n myproject -y && conda activate myproject  # or your preferred environment tool
       (myproject) $ make install

#. Start working on a new feature.

    .. code-block:: console

       $ ballet start-new-feature

    This will create a new Python module within the project's "contrib" directory to hold your
    feature.

    * The contrib directory is usually named something like ``src/ballet_project/features/contrib``.
    * The new subpackage must be named like ``user_<github username>``.
    * The new submodule that will contain the feature must be named like ``feature_<feature name>.py``.

#. Write your feature. We call the code you write to extract one group of related feature values
   a *logical feature*. Within your feature submodule, you can write arbitrary Python code.
   Ultimately, a single object that is an instance of ``ballet.Feature`` must be defined; it will
   be imported by the feature engineering pipeline.

   In this example, a feature is defined that will receive column ``'A'`` from the data and passes
   it through unmodified.

    .. code-block:: python

       import ballet.eng
       from ballet import Feature

       input = 'A'
       transformer = ballet.eng.misc.IdentityTransformer()
       feature = Feature(input=input, transformer=transformer)


#. Submit your feature. Commit your changes and create a pull request to the project repository.

.. figure:: https://upload.wikimedia.org/wikipedia/en/f/f8/Internet_dog.jpg
   :width: 300
   :alt: "On the internet, nobody knows you're a dog" cartoon

   Image from *The New Yorker* cartoon by Peter Steiner, 1993, via Wikipedia.

.. _ballet CLI: https://hdi-project.github.io/ballet/installation.html
