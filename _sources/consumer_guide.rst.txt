==============
Consumer Guide
==============

Do you want to use the feature engineering pipeline developed by a Ballet project team? This guide is for you. It addresses how to install an existing Ballet project for further use, how to use the project's API to engineer features, and how to incorporate a project into a larger ML pipeline.

As a Ballet project repository fills with features, its feature engineering pipeline is always available to engineer features from new data points or datasets. Given the feature testing instituted in the CI/CD for a Ballet project, the latest commit is sure to extract high quality features for your raw data.

Install a project
-----------------

Install a project directly from GitHub with pip using `PEP 508 <https://www.python.org/dev/peps/pep-0508/>`__ specifiers:

.. code-block:: console

   $ pip install 'ballet_predict_house_prices @ git+https://git@github.com/HDI-Project/ballet-predict-house-prices@master'

The project name (before the ``@``) will be different for each project and is shown in the ``name`` key of ``setup.py``. You can substitute ``master`` to a specific git revision like a commit SHA or tag name if you want to install a specific version of the Ballet project. And if the given project actually makes releases to an index like PyPI, you should follow the project's own documentation for how to install.

Next confirm that your install was successful:

.. code-block:: console

   $ python -m ballet_predict_house_prices --help
   Usage: python -m ballet_predict_house_prices [OPTIONS] COMMAND [ARGS]...

   Options:
     --help  Show this message and exit.

   Commands:
     engineer-features  Engineer features

Engineer features
-----------------

Each project exposes two ways to engineer features, using a command line tool (simplest) or using the project as a library (most flexible).

Using the command line
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

   $ python -m myproject engineer-features path/to/test/data path/to/features/output

This will read input data from the given directory, detecting the appropriate entities and targets data files according to the project's configuration. It will load the default development dataset to fit the feature engineering pipeline and target encoder. Using those learned parameters, it will then engineer features on the test dataset and encode the test dataset target. The resulting objects (usually of type ``np.ndarray``) will be saved to the given output directory as ``features.pkl`` and ``target.pkl``.

You can also pass the option ``--train-dir path/to/train/data`` to specify a different training dataset.

For full details, run:

.. code-block:: console

   $ python -m myproject engineer-features --help

Using the library
^^^^^^^^^^^^^^^^^

Import the project's API:

.. code-block:: python

   from myproject.api import api

The API object is an instance of :py:class:`~ballet.project.FeatureEngineeringProject`, providing easy access to the data loading method, feature engineering pipeline, and other functionality implemented by the given Ballet project.

Load the training dataset and a test dataset:

.. code-block:: python

   X_df_tr, y_df_tr = api.load_data()
   X_df_te, y_df_te = api.load_data(input_dir='path/to/test/data')

Fit the feature engineering pipeline:

.. code-block:: python

   result = api.engineer_features(X_df_tr, y_df_tr)
   pipeline, encoder = result.pipeline, result.encoder

The ``engineer_features`` method encapsulates fitting the feature engineering pipeline and target encoder on the given dataset, and also engineering features and encoding the target of the given dataset. The result is an instance of :py:class:`~ballet.pipeline.EngineerFeaturesResult`.

Engineer features on the test dataset:

.. code-block:: python

   X_te = pipeline.transform(X_df_te)

Encode targets of the test dataset:

.. code-block:: python

   y_te = encoder.transform(y_df_te)

You can now use these ``X`` and ``y`` as inputs to your own ML modeling efforts.

To engineer features as a dataframe rather than an array, manually provide a row and column index for the feature matrix:

.. code-block:: python

   X = pd.DataFrame(
       pipeline.transform(X_df),
       columns=pipeline.transformed_names_,
       index=X_df.index,
   )


Build an ML pipeline
--------------------

Ballet integrates with the `MLBlocks`_ library from the `MLBazaar`_ framework. This library supports creating ``MLPipeline`` objects that are generalizations of typical supervised learning pipelines. Typical pipelines from libraries like scikit-learn compose a sequence of transformers and estimators that implement a fit/transform interface. Pipelines from MLBlocks allow arbitrary dataflow between pipeline steps using a shared-memory "context" and support loading "ML primitives" (re-usable ML components from many different libraries) from public catalogs into pipeline steps.

Ballet offers two different ML primitives which can be used in ML pipelines:

- :ref:`mlp_reference:ballet.engineer_features`: this applies the feature engineering pipeline
- :ref:`mlp_reference:ballet.encode_target`: this encodes the target during training and does nothing at prediction time.

Ballet offers two ML pipelines which can be used to make predictions from raw data using Ballet together with an off-the shelf estimator (a random forest regressor or classifier).

- :ref:`mlp_reference:ballet_rf_classifier`
- :ref:`mlp_reference:ballet_rf_regressor`

Installation
^^^^^^^^^^^^

To install MLBlocks, see `here <https://mlbazaar.github.io/MLBlocks/getting_started/install.html>`__ or just run ``pip install mlblocks``. MLBlocks is the only required dependency to loading the ML primitives or creating your own ``MLPipeline``.

If you want to use the pre-defined ``ballet_rf_classifier`` and ``ballet_rf_regressor`` pipelines, they both use a random forest primitive that is defined in the `MLPrimitives`_ catalog. There are two options to be able to use these pre-defined pipelines.

1. You can `install MLPrimitives directly <https://mlbazaar.github.io/MLPrimitives/readme.html#installation>`__, but beware that this will install all dependencies for all primitives in that catalog, including heavyweight dependencies like TensorFlow.

2. You can "install" just the necessary primitives by downloading the primitive annotation to a local catalog:

   .. code-block:: console

      $ mkdir -p ./mlprimitives
      $ curl -O https://raw.githubusercontent.com/MLBazaar/MLPrimitives/master/mlprimitives/primitives/sklearn.ensemble.RandomForestClassifier.json
      $ curl -O https://raw.githubusercontent.com/MLBazaar/MLPrimitives/master/mlprimitives/primitives/sklearn.ensemble.RandomForestRegressor.json

ML pipeline usage
^^^^^^^^^^^^^^^^^

In this example, we use the project's API to load data and then fit an ML pipeline. Under the hood, the ML pipeline is using the two ML primitives that Ballet provides to engineer features from the raw data and encode the target.

.. code-block:: python

   from myproject.api import api
   from mlblocks import MLPipeline, load_pipeline

   X_df, y_df = api.load_data()
   X_df_te, y_df_te = api.load_data(input_dir='path/to/test/data')

   pipeline = MLPipeline(load_pipeline('ballet_rf_classifier'))

   pipeline.fit(X_df, y_df)
   y_pred = pipeline.predict(X_df)

   y_pred_te = pipeline.predict(X_df_te)

Ballet project detection
^^^^^^^^^^^^^^^^^^^^^^^^

Ballet needs to be able to detect the Ballet project we are working with in order to access it's API. In most cases, this is done automatically, as Ballet ascends the file system from the current working directory looking for a project root. In other cases, you may need to specify the details of the project manually.

The same principles apply when using an MLPipeline. If Ballet cannot automatically detect the project, you have to help it out. You can do this by either specifying the name of the package or the path to the project. If you have downloaded the project using ``pip install`` as above, then the name of the package is a better approach because you shouldn't need to care where pip has copied source files.

You have to specify ``init_params`` of both ML primitives:

.. code-block:: python

   pipeline = MLPipeline(
       pipeline=load_pipeline('ballet_rf_classifier'),
       init_params={
          'ballet.engineer_features#1': {
              'package_slug': 'myproject',
          },
          'ballet.encode_target#1': {
              'package_slug': 'myproject',
          },
       }
   )

See the documentation of the adapter functions for more details: :py:func:`~ballet.mlprimitives.make_engineer_features` and :py:func:`~ballet.mlprimitives.make_encode_target`.

Primitive and pipeline discovery
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you install a Ballet project, Ballet is installed as a dependency. Ballet exposes its ML primitives and ML pipelines using entry points so they are automatically discoverable by MLBlocks and can be loaded with ``load_pipeline`` and ``load_primitive``.

You can also list all of the Ballet primitives and pipelines:

.. code-block:: python

   from mlblocks import find_primitives, find_pipelines
   find_primitives(pattern='ballet.*')
   find_pipelines(pattern='ballet.*')

.. _MLBlocks: https://mlbazaar.github.io/MLBlocks/
.. _MLPrimitives: https://mlbazaar.github.io/MLBlocks/
.. _MLBazaar: https://mlbazaar.github.io
