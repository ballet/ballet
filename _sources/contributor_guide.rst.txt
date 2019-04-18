=================
Contributor Guide
=================

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
