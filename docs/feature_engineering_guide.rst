=========================
Feature Engineering Guide
=========================

Feature engineering is the process of transforming raw variables into
feature values that can be input to a learning algorithm. We include every step that is needed to go from the raw dataset to the learning algorithm: cleaning missing values and outliers, scaling values, deriving complex features from multiple variables, reducing dimensionality, encoding categorical and ordinal variables, and more.

In Ballet, feature engineering is centered around creating feature definitions.
These are modular, flexible, and expressive and will allow us to compose an
entire feature engineering pipeline out of individual feature objects.

Loosely, a feature is a tuple ``(input_columns, transformer_to_apply)``.
Your job in feature engineering will be to define this tuple of input
columns and a transformer to apply on them.

.. tip::

   By the end of this guide, you will

   #. understand the concept of a "feature definition"
   #. be able to write a simple ``Feature`` in Ballet
   #. be familiar with the feature engineering primitives provided in the :py:mod:`ballet.eng` package

Feature Definitions
-------------------

A feature definition (or simply "feature") is the semantics and implementation of code to extract feature values from raw data. It is a learned map from raw variables in one data instance to feature values.

Less formally, a feature has

- a meaning, like "column 1 after it has been cleaned using information from column 2"
- a code representation, like a Python object that takes as input rows of raw data and produces as output rows of feature values. It also has a separate stage to *learn* import parameters from the rows of training data before it can be applied to the training data or to unseen test data.

A feature can produce either

- a scalar feature value for each instance
- a vector of feature values, as in the case of the embedding of a categorical variable.

Each feature is "parameterized" by a dataset, usually the training dataset, indicating that it learns any information it uses, such as variable means and variances. This formalizes the separation between training and testing data to avoid any "leakage" of information during the feature engineering process.

In Ballet, features are realized in Python as instances of :py:class:`~ballet.feature.Feature` with the following attributes:

- ``input``: the input to the feature, in terms of columns of the raw dataset.
- ``transformer``: the transformation applied to the raw data. The transformer is an object (or sequence of objects) that provide (or each provide) a fit/transform interface.

Why?
^^^^

In the data science community, it is common to do feature engineering by applying a sequence of
mutations to a data frame object or using ``sklearn.preprocessing`` objects. Why do we go through
hoops to use :py:class:`~ballet.feature.Feature` objects?

#. *Modularity.* Each feature stands alone and can be reasoned about,
   validated, and implemented separately.

#. *Avoid leakage.* By writing all features as learned transformations (with
   separate fit and transform stages) and enforcing a train-test split, we
   ensure that feature engineering code never sees test data before it applies
   transformations on new instances.

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

Writing features
----------------

Let's dive right into writing features.

A first example
^^^^^^^^^^^^^^^

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

This feature requests one input, the ``Lot Area`` column of the raw dataset. It's transformer is given as ``None``, a value indicating the identity transformation. (This is equivalent to providing ``lambda df: df`` or :py:class:`ballet.eng.IdentityTransformer`.) Thus when the feature is executed as part of a pipeline, the transformer's fit
and transform methods will be called receiving the one column as input.

.. code-block:: python

   pipeline = feature.as_feature_engineering_pipeline()  # type: ballet.pipeline.FeatureEngineeringPipeline
   pipeline.fit_transform(X_df)
   # array([[31770],
   #        [11622],
   #        [14267],
   #        [11160],
   #        [13830]])

The behavior is similar to the following "imperative" code:

.. code-block:: python

   x = X_df[input]
   transformer.fit_transform(x)

A second example
^^^^^^^^^^^^^^^^

Let's take a look at a slightly more complex example.

.. include:: fragments/feature-engineering-guide-second-feature.py
  :code: python

The feature requests one input, the ``Lot Frontage`` column. Its transformer is a :py:class:`~ballet.eng.sklearn.SimpleImputer` instance. It will receive as its input a DataFrame with one column. So we get the following behavior when the feature is executed as part of a pipeline:

- during training (fit-stage), the transformer will compute the mean of the ``Lot Frontage`` values that are not missing
- during training (transform-stage), the transformer will replace missing values with the computed training mean
- during testing (transform-stage), the transformer will replace missing values with the computed training mean

The :py:class:`~ballet.eng.sklearn.SimpleImputer` is re-exported by Ballet from scikit-learn; see the full assortment of transformers that are available in :py:mod:`ballet.eng`.

A third example
^^^^^^^^^^^^^^^

Let's took a look at another example.

.. include:: fragments/feature-engineering-guide-third-feature.py
   :code: python

The feature requests three inputs, which are various measures of square footage in the house (basement, first floor, and second floor). The combined transformer is a sequence of two "transformer-likes". The first transformer in is a function that will receive as its input a DataFrame with three columns, and it sums across rows (``axis=1``), returning a single column with the total square footage. The second transformer is a utility object that replaces missing values. In this case, neither transformer learns anything from data (i.e. it does not need to save parameters learned from the training data) so both can be simple functions. Here, the first function is implicitly converted into a :py:class:`~ballet.eng.sklearn.FunctionTransformer` and the second transformer is already a thin wrapper around ``pd.fillna``.

In this feature, the sum is equivalent to a weighted sum with the weights all equal to 1. But maybe you have the intuition that not all living area is created equal? You might apply custom weights as follows:

.. code-block:: python

   lambda df: 0.5 * df["Total Bsmt SF"] + 1 * df["1st Flr SF"] + 2 * df["2nd Flr SF"]

Regardless, we should note that the model may be able to *learn* these weights, even if it is a simple linear regression. Thus another good feature may be one that just cleans these three columns of missing values and possible scales them.

.. include:: fragments/feature-engineering-guide-third-feature-v2.py
   :code: python

Your job is to think best about what the model can learn from different features and be creative about how you can apply your expert intuition to the problem. You could submit both of these features and leave it up to the feature validation stage to accept one or both of these based on their performance.

Input types and conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``input`` field accepts either a key or a collection of keys (usually strings) identifying
columns from the raw data.

- if ``input`` is a scalar key, a 1-dimensional pandas Series or numpy array is passed to the
  transformer
- if ``input`` is a collection of keys, a 2-dimensional pandas DataFrame or numpy array is
  passed to the transformer

With respect to the discussion about robustness above, Ballet tries to pass the most obvious
objects to the transformer. For example, if the raw data is a pandas ``DataFrame`` and ``input``
is a scalar key, Ballet tries to pass a ``Series`` to the transformer. If that fails in a
predictable way (i.e. the transformer appears to not be able to handle that data type), then Ballet
tries again with the next most obvious input data type (a 1-d numpy array), continuous on to a
pandas ``DataFrame`` with one column and finally a 2-d numpy array with one column. The same
principles apply when ``input`` is a collection of keys, except Ballet will not try to pass any 1-d
data.

Transformers
^^^^^^^^^^^^

The ``transformer`` field accepts either one or a list of transformer-like objects.

A *transformer-like* is any of the following:

- an object that satisfies the scikit-learn `Transformer API`_, having ``fit``, ``transform``, and ``fit_transform`` methods.
- a callable that accepts the ``X`` DataFrame as input and produces an array-like as output. This can be thought of as a transformer that does not have a fit stage. Ballet will take care of converting it into a :py:class:`~ballet.eng.sklearn.FunctionTransformer` object.
- the value ``None``, shorthand to indicate the identity transformer. Ballet will convert it into a :py:class:`~ballet.eng.IdentityTransformer` object.

Feature engineering pipelines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A feature engineering pipeline can be created out of a collection of Features. It has a
fit/transform API. When applied to raw data, it applies each underlying feature in parallel,
concatenating the results.

Feature engineering primitives
------------------------------

Many features exhibit common patterns, such as scaling or imputing variables using simple
procedures. And while some features are relatively simple and have no learning component, others
are more involved to express. Commonly, data scientists extract these more advanced features by
manipulating training and test tables directly using popular libraries like *pandas* or *dplyr*
(often leading to leakage), whereas these operations should instead be rewritten in a fit/transform
style.

To ease this process, Ballet provides a library of feature engineering primitives,
:py:mod:`ballet.eng`, which implements many common learned transformations and utilities.

Operating on groups
^^^^^^^^^^^^^^^^^^^

See:

- :py:class:`ballet.eng.GroupedFunctionTransformer`
- :py:class:`ballet.eng.GroupwiseTransformer`
- :py:class:`ballet.eng.ConditionalTransformer`

Addressing missing values
^^^^^^^^^^^^^^^^^^^^^^^^^

See:

- :py:mod:`ballet.eng.missing`

Operating on time series data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

See :py:mod:`ballet.eng.ts`

Other primitives
^^^^^^^^^^^^^^^^

See:

- :py:class:`ballet.eng.SimpleFunctionTransformer`
- :py:class:`ballet.eng.ConditionalTransformer`

External libraries
^^^^^^^^^^^^^^^^^^

Many feature engineering primitives are also available in scikit-learn and other libraries. Don't reinvent the wheel!

Ballet re-exports feature engineering primitives from external libraries. Note that not all primitives may be relevant for all projects, for example many feature engineering primitives from ``skits`` and ``tsfresh`` are only appropriate for time-series forecasting problems.

- :py:mod:`ballet.eng.category_encoders` (primitives from `category_encoders`_)
- :py:mod:`ballet.eng.feature_engine` (primitives from `feature_engine`_)
- :py:mod:`ballet.eng.featuretools` (primitives from `featuretools`_)
- :py:mod:`ballet.eng.skits` (primitives from `skits`_)
- :py:mod:`ballet.eng.sklearn` (primitives from `sklearn`_)
- :py:mod:`ballet.eng.tsfresh` (primitives from `tsfresh`_)

All of these are further re-exported in the catch-all ``external`` module:

- :py:mod:`ballet.eng.external` (all external primitives)

Pandas ⇔ Ballet Examples
------------------------

It may be helpful to see a lot of examples of pandas-style code with the associated Ballet feature implementations, especially if you are more familiar with writing imperative pandas code.

Recall that we will be working with the simplified table ``X_df`` `from above <#writing-features-x-df>`__.

The first two examples are pretty simple.

In example 3, we can see where Ballet's ``Feature`` can be helpful compared to vanilla pandas for machine learning: we can easily incorporate a transformer from scikit-learn that learns the column mean on the training data and applies it on unseen test data.

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

As you come up with more creative features, you may find that you need to create your own
transformer classes. Here are some tips for creating your own transformers.

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

Let's create a feature that captures whether a string variable is the longest value observed in the
data. This is a silly feature for a number of reasons, so don't take it too seriously, but it
demonstrates the steps required to roll your own transformer.

.. code-block:: python

   from ballet import Feature
   from ballet.eng import BaseTransformer

   input = 'Exterior 1st'

   class LongestStringValue(BaseTransformer):

       def fit(self, X, y=None):
           self.longest_string_length_ = X.str.len().max()
           return self

       def transform(self, X):
           return X.str.len() >= self.longest_string_length_

   transformer = LongestStringValue()
   feature = Feature(input=input, transformer=transformer)

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
