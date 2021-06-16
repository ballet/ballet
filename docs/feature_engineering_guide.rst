=========================
Feature Engineering Guide
=========================

Feature engineering is the process of transforming raw variables into
feature values that can be input to a learning algorithm. We consider in this process every step that is needed to go from the raw dataset to the learning algorithm: cleaning missing values and outliers, scaling values, deriving complex features from multiple variables, reducing dimensionality, encoding categorical and ordinal variables, and more.

In Ballet, feature engineering is centered around creating feature definitions.
These are modular, flexible, and expressive and will allow us to compose an
entire feature engineering pipeline out of individual feature objects.

.. tip::

   By the end of this guide, you will

   #. understand the concept of a "feature definition"
   #. be able to write feature definitions in Ballet
   #. be familiar with the feature engineering primitives provided in the :py:mod:`ballet.eng` package

Feature engineering in 90 seconds
---------------------------------

A *feature definition*, or simply "feature," is a tuple ``(input_columns, transformer_to_apply)``.  Your job in feature engineering is to define this tuple of input columns and a transformer to apply on them.

A feature "requests" an input column or a set of input columns. Then it defines a transformation to apply to this input. This *transformer* may be one step or a list of steps. Each step provides a *fit* method and a *transform* method, though Ballet supports more concise syntax for the most common transformations. When features are part of a *feature engineering pipeline*, their outputs are concatenated together to form a *feature matrix*.

Let's take a look at the simplest possible feature:

.. code-block:: python

   from ballet import Feature
   input = "Lot Area"
   transformer = None
   feature = Feature(input, transformer)

We have to create an instance of :py:class:`~ballet.feature.Feature`. The special value ``None`` will be replaced by Ballet with an object that applies that identity transformation. This feature, when applied to a dataset that contains the column ``Lot Area``, will pass that column through unchanged. It's not super exciting.

Let's look at a more realistic feature:

.. include:: fragments/feature-engineering-guide-third-feature.py
   :code: python

This feature requests three input columns and then defines two transformer steps. The first step subtracts the area of a house's garage and first floor from its lot. The second step fills in missing values with the median of the training dataset. When applied to a dataset that contains the three columns, this feature will compute some measure of "yard area" for a house.

You have enough to get started! But read on to learn all about the powerful functionality that Ballet provides for developing feature definitions.

Writing features
----------------

Feature definitions
^^^^^^^^^^^^^^^^^^^

A feature definition, or simply "feature," is the code to extract feature values from raw data, paired with meta-information about the transformation.

In Ballet, features are realized in Python as instances of :py:class:`~ballet.feature.Feature` with the following attributes:

- ``input``: the input columns to the feature from the raw dataset.
- ``transformer``: the transformation applied to the selected columns. The transformer is an object (or sequence of objects) that provide (or each provide) ``fit`` and ``transform`` methods.
- ``name``: the name of the feature.
- ``description``: a longer, human-readable description of the feature.

A feature can either be scalar-valued (produce a scalar feature value for each data instance) or it can be vector-valued (produce a vector of feature values for each data instance, as in the embedding of a categorical variable). It learns parameters from a training dataset which it can then use to extract feature values from previously unseen data instances, avoiding any "leakage" of information through feature engineering.

Input types
^^^^^^^^^^^

The ``input`` field accepts either a key or a collection of keys (usually strings) identifying columns from the raw data.

- if ``input`` is a scalar key, a 1-dimensional pandas Series or numpy array is passed to the transformer
- if ``input`` is a collection of keys, a 2-dimensional pandas DataFrame or numpy array is passed to the transformer

There is also experimental support for some other ways of indexing data frames, such as `selection by callable <https://pandas.pydata.org/docs/user_guide/indexing.html#selection-by-callable>`__. Here, ``input`` must be a function that accepts one argument that returns valid output for indexing, such as ``lambda df: ['A', 'B']``. This is not officially supported by the underlying sklearn-pandas library, so please report any issues you experience.

.. versionchanged:: 0.19

Transformers
^^^^^^^^^^^^

The ``transformer`` field accepts one or more transformer steps.

A *transformer step* is a *transformer-like object* that satisfies the scikit-learn `Transformer API`_, having ``fit``, ``transform``, and ``fit_transform`` methods.

In addition, Ballet supports a more concise syntax for expressing common operations. When a feature is created, it replaces this shorthand with explicit transformers. The following syntax is supported:

- a callable that accepts the ``X`` DataFrame as input and produces an array-like as output. This can be thought of as a transformer that does not have a fit stage. Ballet will convert it into a :py:class:`~ballet.eng.sklearn.FunctionTransformer` object.
- the value ``None``, shorthand to indicate the identity transformer. Ballet will convert it into an :py:class:`~ballet.eng.IdentityTransformer` object.
- a tuple of ``(input, transformer)`` to indicate a nested transformation. This transformer then operates on only a subset of the inputs that your feature is already working on. Both elements of the tuple are interpreted the same as if they were passed to the :py:class:`~ballet.feature.Feature` constructor. Internally, they will be converted to a :py:class:`~ballet.eng.base.SubsetTransformer`, which you can also use directly.
- another feature instance itself! This is another way to nest transformations. You can import a feature from another module and use it within your own transformer.

Automatic data conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^

Ballet wraps each transformer step with additional code in order to make it more robust to different data types. It does this by trying to pass the most "obvious" data types to the transformer. If the transformation fails in a predictable way (i.e. the transformer appears to not be able to handle that data type), then Ballet tries again with the next most obvious input data type.

If the input is a scalar key, Ballet tries to pass, in order:

#. a ``Series``
#. a 1-d numpy array (vector)
#. a ``DataFrame`` with one column
#. a 2-d numpy array

If the input is a collection of keys, Ballet tries to pass, in order:

#. a ``DataFrame``
#. a 2-d numpy array

Usually this process will help you behind the scenes by catching and resolving errors due to incorrect data types. However, if the transformer fails on *all* data types, then Ballet gives up. Each error message from each data conversion attempt will be shown together. For debugging, you should start by looking at the top most error message. You can also enable the Ballet logger at the debug level (``ballet.util.log.enable('ballet', level='DEBUG')``) to see the details of the conversion attempts. You may see :py:class:`~ballet.transformer.DelegatingRobustTransformer` in the stacktrace, which does the wrapping. We are working on improving the debugging experience in this situation.

Feature engineering pipelines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A feature engineering pipeline can be created out of zero or more features. It has a fit/transform API. When applied to raw data, it applies each underlying feature in parallel and concatenates the results.

Okay, but why?
^^^^^^^^^^^^^^

In the data science community, it is common to do feature engineering by applying a sequence of
mutations to a data frame object or using ``sklearn.preprocessing`` objects. Why do we go through
hoops to use :py:class:`~ballet.feature.Feature` objects?

#. *Modularity.* Each feature stands alone and can be reasoned about,
   validated, and implemented separately.

#. *Leakage.* By writing all features as learned transformations (with
   separate fit and transform stages) and enforcing a train-test split, we
   ensure that feature engineering code never sees test data before it applies
   transformations on new instances, helping better estimate generalization performance.

#. *Clearly declare inputs and outputs.* Each feature declares its own inputs
   (and optionally outputs) and can operate on them only. Thus a feature can
   impute missing values in a single column, as opposed to the entire dataset,
   and different ``Feature`` objects can target different subsets of the input
   variable space.

#. *Pipelines.* Feature objects can be easily composed together can be
   combined into a pipeline that can learn feature transformations from
   training data and apply them on new instances.

#. *Robustness.* Data scientists are often surprised to find the number of
   errors that arise from trying to use multiple libraries together, such as
   pandas and scikit-learn. Common errors include scikit-learn transformers and
   estimators failing on columnar data that has the wrong number of dimensions
   (i.e. 1-dimensional or 2-dimensional column vectors). Features in Ballet
   magically transform feature input data appropriately to avoid common errors.

Learn by example
----------------

Let's dive into some examples.

A simple feature
^^^^^^^^^^^^^^^^

Here is a simple feature, for the `HDI-Project/ballet-predict-house-prices
<https://github.com/HDI-Project/ballet-predict-house-prices>`_ problem. Imagine we are working with
the following simplified dataset:

.. include:: fragments/simple_table.py
   :code: python

So ``X_df`` looks as follows (the blank cells are ``NaN`` values):

.. csv-table:: ``X_df``
   :name: writing-features-x-df
   :header-rows: 1
   :file: fragments/simple_table_data.csv
   :delim: tab

We define our first feature:

.. include:: fragments/feature-engineering-guide-first-feature.py
   :code: python

This feature requests one input, the ``Lot Area`` column of the raw dataset.  It's transformer is given as ``None``, syntactic sugar indicating the identity transformation. When the feature is executed as part of a pipeline, the transformer's fit and transform methods will be called receiving the one column as input. Here we access the ``pipeline`` property, which is a convenience to show a feature engineering pipeline containing just the one feature.

.. code-block:: python

   pipeline = feature.pipeline
   pipeline.fit_transform(X_df, y=y_df)
   # array([[31770],
   #        [11622],
   #        [14267],
   #        [11160],
   #        [13830]])

The behavior is similar to the following "imperative" code:

.. code-block:: python

   x = X_df[input]
   transformer.fit_transform(x, y=y_df)

Learned transformations
^^^^^^^^^^^^^^^^^^^^^^^

Let's take a look at another example.

.. include:: fragments/feature-engineering-guide-second-feature.py
  :code: python

The feature requests one input, the ``Lot Frontage`` column. Its transformer is a :py:class:`~ballet.eng.external.SimpleImputer` instance. It will receive as its input a data frame with this one column. So we get the following behavior when the feature is executed as part of a pipeline:

- during training (fit-stage), the transformer will compute the mean of the ``Lot Frontage`` values that are not missing
- during training (transform-stage), the transformer will replace missing values with the computed training mean
- during testing (transform-stage), the transformer will replace missing values with the computed training mean

:py:class:`~ballet.eng.external.SimpleImputer` is re-exported by Ballet from scikit-learn; see the full assortment of primitives that are available `below <#feature-engineering-primitives>`__.

Multi-column inputs and transformer lists
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's took a look at another example.

.. include:: fragments/feature-engineering-guide-third-feature.py
   :code: python

The feature requests three inputs, which are various measures of square footage on the property. The combined transformer is a sequence of two transformer steps. The first transformer step is a function that will receive as its input a DataFrame with three columns, and it computes the portion of the lot area that is not attributable to the garage or the house footprint. However, there may be missing values after this operation if any of the columns had missing values, so we impute them with the median of the training dataset. Here, the first function is implicitly converted into a :py:class:`~ballet.eng.FunctionTransformer` (other shorthand is also supported for `specifying transformer steps <#transformers>`__).

.. In this feature, the sum is equivalent to a weighted sum with the weights all equal to 1. But maybe you have the intuition that not all living area is created equal? You might apply custom weights as follows:

.. .. code-block:: python

..    lambda df: 0.5 * df["Total Bsmt SF"] + 1 * df["1st Flr SF"] + 2 * df["2nd Flr SF"]

.. Regardless, we should note that the model may be able to *learn* these weights, even if it is a simple linear regression. Thus another good feature may be one that just cleans these three columns of missing values and possible scales them.

.. .. include:: fragments/feature-engineering-guide-third-feature-v2.py
..    :code: python

.. Your job is to think best about what the model can learn from different features and be creative about how you can apply your expert intuition to the problem. You could submit both of these features and leave it up to the feature validation stage to accept one or both of these based on their performance.

Nested transformers
^^^^^^^^^^^^^^^^^^^

You can even nest transformers or other features within the ``transformer`` field. This next feature calculates how "narrow" a certain house's lot is by measuring the ratio between the frontage of the lot and its depth, and scaling the resulting values to a 0-1 range.

.. include:: fragments/feature-engineering-guide-fourth-feature.py
   :code: python

The first transformer step shows how to use nested transformers. You can pass a tuple of ``(input, transformer)``, just like the current feature you are developing! Here, we are re-using the same input and transformer from the ``impute_lot_frontage`` that we `already developed <#example-learned-transformation>`__. Any columns that are not used in the nested transformer will be passed through to the next transformer step unchanged. The next transformer step computes the ratio of depth (area/frontage) to frontage; a value of 1.0 is a square lot, a larger value is a more narrow lot. Since a smaller value is a less narrow lot, for the purpose of this feature we clip these smaller values to 1. There are a lot of squarish lots, so we also take the log of this ratio to address skew. Finally, we scale the measure to a 0-1 range.

Complex features
^^^^^^^^^^^^^^^^

Here's another example of a feature that is a bit more complex. The idea here is that we want to encode ``Exterior 1st`` (exterior covering on house) but first we have to impute missing values. Since houses in a neighborhood may have similar exterior characteristics, we can impute missing values with the most common exterior covering in the neighborhood.

To do this we have to first determine the most common exterior covering per neighborhood in the training dataset, then store that mapping. This is where Ballet's :py:class:`~ballet.eng.GroupwiseTransformer` comes in handy, which clones the transformer many times and fits it separately for each group. Finally we can encode the categories using a :py:class:`~ballet.eng.external.OneHotEncoder` for example.

.. include:: fragments/feature-engineering-guide-fifth-feature.py
   :code: python

Feature engineering primitives
------------------------------

Many features exhibit common patterns, such as scaling or imputing variables using simple procedures. And while some features are relatively simple and have no learning component, others are more involved to express. Commonly, data scientists extract these more advanced features by manipulating training and test tables directly using popular libraries like *pandas* or *dplyr* (often leading to leakage), whereas these operations should instead be rewritten in a fit/transform style.

To ease this process, Ballet provides a library of feature engineering primitives, :py:mod:`ballet.eng`, which implements many common learned transformations and utilities.

Operating on groups
^^^^^^^^^^^^^^^^^^^

- :py:class:`ballet.eng.GroupedFunctionTransformer`
- :py:class:`ballet.eng.GroupwiseTransformer`
- :py:class:`ballet.eng.ColumnSelector`
- :py:class:`ballet.eng.SubsetTransformer`

Logic to avoid leakage
^^^^^^^^^^^^^^^^^^^^^^

- :py:class:`ballet.eng.ConditionalTransformer`
- :py:class:`ballet.eng.ComputedValueTransformer`

Data cleaning
^^^^^^^^^^^^^

- :py:class:`ballet.eng.LagImputer`
- :py:class:`ballet.eng.NullFiller`
- :py:class:`ballet.eng.NullIndicator`
- :py:class:`ballet.eng.ValueReplacer`

Operating on time series data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :py:class:`ballet.eng.SingleLagger`
- :py:func:`ballet.eng.make_multi_lagger`

Utilities
^^^^^^^^^

- :py:class:`ballet.eng.NamedFramer`
- :py:class:`ballet.eng.SimpleFunctionTransformer`

Other primitives
^^^^^^^^^^^^^^^^

- :py:class:`ballet.eng.BoxCoxTransformer`

External libraries
^^^^^^^^^^^^^^^^^^

Many feature engineering primitives are also available in scikit-learn and other libraries. Don't reinvent the wheel!

Ballet re-exports feature engineering primitives from external libraries. Note that not all primitives may be relevant for all projects, for example many feature engineering primitives from ``skits`` and ``tsfresh`` are only appropriate for time-series forecasting problems.

All of these are further re-exported in the catch-all ``external`` module, which is the best place to import from in your code:

- :py:mod:`ballet.eng.external`: all external primitives

You can also browse the list of primitives from each external library:

- :py:mod:`ballet.eng.external.category_encoders`: primitives from `category_encoders`_
- :py:mod:`ballet.eng.external.feature_engine`: primitives from `feature_engine`_
- :py:mod:`ballet.eng.external.featuretools`: primitives from `featuretools`_
- :py:mod:`ballet.eng.external.skits`: primitives from `skits`_
- :py:mod:`ballet.eng.external.sklearn`: primitives from `sklearn`_
- :py:mod:`ballet.eng.external.tsfresh`: primitives from `tsfresh`_

Differences from Pandas
-----------------------

It may be helpful to see a lot of examples of pandas-style code with the associated Ballet feature implementations, especially if you are more familiar with writing "imperative" pandas code.

Recall that we will be working with the simplified table ``X_df`` `from above <#writing-features-x-df>`__.

The first two examples are pretty simple.

In example 3, we can see where Ballet's :py:class:``~ballet.feature.Feature`` can be helpful compared to vanilla pandas for machine learning: we can easily incorporate a transformer from scikit-learn that learns the column mean on the training data and applies it on unseen test data.

In example 4, we see how multiple transformers can be chained together in a list.


.. list-table::
   :width: 100%
   :header-rows: 1

   * -
     - Pandas
     - Ballet
   * - 1
     - .. include:: fragments/pandas-ballet-ex1-pandas.py
          :code: python
     - .. include:: fragments/pandas-ballet-ex1-ballet.py
          :code: python
   * - 2
     - .. include:: fragments/pandas-ballet-ex2-pandas.py
          :code: python
     - .. include:: fragments/pandas-ballet-ex2-ballet.py
          :code: python
   * - 3
     - .. include:: fragments/pandas-ballet-ex3-pandas.py
          :code: python
     - .. include:: fragments/pandas-ballet-ex3-ballet.py
          :code: python
   * - 4
     - .. include:: fragments/pandas-ballet-ex4-pandas.py
          :code: python
     - .. include:: fragments/pandas-ballet-ex4-ballet.py
          :code: python

Rolling your own transformers
-----------------------------

As you come up with more creative features, you may find that you need to create your own transformer classes. Here are some tips for creating your own transformers.

#. Build off of :py:class:`ballet.eng.BaseTransformer` which inherits from :py:class:`sklearn.base.BaseEstimator`, :py:class:`sklearn.base.TransformerMixin`, and :py:class:`ballet.eng.NoFitMixin`.

#. (Optional) Implement the ``__init__`` method::

      def __init__(self, **kwargs)

   This method is optional if your transformer does not have any hyperparameters. Following the scikit-learn convention, the init method should take keyword arguments only and do nothing more then set them on ``self``. Each keyword argument is a hyperparameter of the transformer.

#. (Optional) Implement the ``fit`` method::

      def fit(self, X, y=None)

   If you do not require a fit method, you can omit this, as a no-op fit method is already provided by the parent class. Any learned parameters should be set on ``self``. Following the scikit-learn convention, the names of learned parameters should have trailing underscores. Finally, ``fit`` should ``return self`` so that it can be chained with other methods on the class::

       self.theta_ = 5
       return self

#. Implement the ``transform`` method::

      def transform(self, X)

   Here you can assume the learned parameters, if any, are available.

You can also read the `scikit-learn documentation on a similar topic <https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator>`__. (Note that this documentation page is likely overkill for the types of transformers you may be implementing.

Example
^^^^^^^

Let's create a feature that captures whether a string variable is the longest value observed in the data. This is a silly feature for a number of reasons, so don't take it too seriously, but it demonstrates the steps required to roll your own transformer.

.. include: fragments/feature-engineering-guide-custom-transformer.py
   :code: python


Okay, let's unpack what happened here. First, we declared the input to this feature, ``'Exterior 1st'``, a scalar key, so the feature will receive a pandas ``Series`` as the input ``X``. Next we created a new class that inherits from ``BaseTransformer``. The transformer does not have any "hyperparameters" so we can skip defining an ``__init__`` method. We learn the ``longest_string_length_`` parameter during the fit stage and set it on ``self``. We were able to assume that ``X`` is a series, and thus has the ``.str`` vectorized string accessor. We can assume this because Ballet will automatically try to pass the input in various formats and will store the format that worked, i.e. "series". (If this were to be a new feature engineering primitive that would be used in more than this one situation, we might want to add logic to allow the feature to operate on a DataFrame as well.) Next, in the transform stage, we check for each new instance whether the length is greater than or equal to the longest string length observed in the training data. The result will be a 1-d array (series) of ``bool``\ s. Finally, having created the transformer class, we create an instance of it and create our ``Feature`` object.

Further reading
---------------

- :py:class:`ballet.feature.Feature`
- :py:class:`ballet.pipeline.FeatureEngineeringPipeline`

.. _`Transformer API`: https://scikit-learn.org/stable/glossary.html#term-transformer
.. _`category_encoders`: http://contrib.scikit-learn.org/category_encoders/
.. _`feature_engine`: https://feature-engine.readthedocs.io/en/latest/
.. _`featuretools`: https://www.featuretools.com/
.. _`skits`: https://github.com/EthanRosenthal/skits
.. _`sklearn`: https://scikit-learn.org/stable/index.html
.. _`tsfresh`: https://tsfresh.readthedocs.io/en/latest/index.html
