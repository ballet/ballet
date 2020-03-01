=========================
Feature Engineering Guide
=========================

Feature engineering is the process of transforming raw variables into feature values that can be
input to a learning algorithm. In Ballet, feature engineering is centered around creating logical
features.

At the end of the day, a feature is a tuple ``(input_columns, transformer_to_apply)``. Your job
in feature engineering will be to define this tuple of input columns and a transformer to apply
on them.

.. tip::

   By the end of this guide, you will

   #. understand the concept of a "logical feature"
   #. be able to write a simple ``Feature`` in Ballet
   #. be familiar with the feature engineering primitives provided in the ``ballet.eng`` package

Logical Features
----------------

A logical feature is the semantics and implementation of code to extract feature values from raw
data.

It is a learned map from raw variables in one data instance to feature values,

.. math::

   f: \mathcal{D} \to \mathcal{X} \to \mathbb{R}^{q_f},

where :math:`q_f` is the dimensionality of the feature values extracted by :math:`f`.

A logical feature can produce either a
scalar feature value for each instance (q = 1) or a vector of feature values, as in the case of the
embedding of a categorical variable (q > 1). Each logical feature is parameterized by D indicating
that it learns any information it uses, such as variable means and variances, from D. This
formalizes the separation between training and testing data to avoid any "leakage" of information
during the feature engineering process.

In Ballet, logical features are realized in Python as objects of type ``ballet.feature.Feature``.
The input to the feature, in terms of columns of the raw dataset, is specified in the ``input``
field. The transformation applied to the raw data is specified in the ``transformer`` field. The
transformer is an object (or sequence of objects) that provide (or each provide) a fit/tranform
interface.

Writing features
----------------

Let's dive right into writing features.

A first example
^^^^^^^^^^^^^^^

Here is a simple feature, for the `HDI-Project/ballet-predict-house-prices
<https://github.com/HDI-Project/ballet-predict-house-prices>`_ problem. Imagine we are working with
the following simplified dataset:

.. code-block:: python

   import pandas as pd
   X_df = pd.DataFrame(data={
      'Lot Frontage': [141, 80, 81, 93, 74],
      'Lot Area': [31770, 11622, 14267, 11160, 13830],
      'Exterior 1st': ['BrkFace', 'VinylSd', 'Wd Sdng', 'BrkFace', 'VinylSd'],
   })
   y_df = pd.Series(data=[215000, 105000, 172000, 244000, 189900], name='Sale Price')


We define our first feature:

.. code-block:: python

   from ballet import Feature
   from ballet.eng.misc import IdentityTransformer

   input = 'Lot Area'
   transformer = IdentityTransformer()
   feature = Feature(input=input, transformer=transformer)

This feature requests one input, the ``Lot Area`` column of the raw dataset. It's transformer is a
simple identity map. Thus when the feature is executed as part of a pipeline, the transformer's fit
and transform methods will be called receiving the one column as input.

.. code-block:: python

   import pandas as pd
   from ballet.pipeline import FeatureEngineeringPipeline
   pipeline = FeatureEngineeringPipeline([feature])
   pipeline.fit_transform(X_df)
   # array([[31770],
   #        [11622],
   #        [14267],
   #        [11160],
   #        [13830]])

The semantics are similar to the following imperative code:

.. code-block:: python

   x = X_df[input].values
   transformer.fit_transform(x)

Why?
^^^^

In the data science community, it is common to do feature engineering by applying a sequence of
mutations to a data frame object or using ``sklearn.preprocessing`` objects. Why do we go through
hoops to use ``Feature`` objects?

#. *Enforce train/test split.* By writing all features as learned transformations (with separate
   fit and transform stages), we ensure that feature engineering code never sees test data before
   it applies transformations on new instances.
#. *Clearly declare inputs and outputs.* Each feature declares its own inputs (and optionally
   outputs) and can operate on them only. Thus a feature can impute missing values in a single
   column, as opposed to the entire dataset, in the case of the scikit-learn ``Imputer`` for
   example.
#. *Facilitate pipeline idiom.* Each feature stands alone but the objects together can be combined
   into a pipeline that can learn feature transformations from training data and apply them on
   new instances.
#. *Add robustness.* Users are often surprised to find the number of errors that arise from trying
   to use multiple libraries together, such as pandas and scikit-learn. Common errors include
   scikit-learn transformers and estimators failing on columnar data that has the wrong number of
   dimensions (i.e. 1-dimensional or 2-dimensional column vectors). Features in Ballet magically
   transform feature input data appropriately to avoid common errors.

Input types and conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``input`` field accepts either a key or a collection of keys (usually strings) identifying
columns from the raw data.

- if ``input`` is a scalar key, a 1-dimensional pandas Series or numpy array is passed to the
  transformer
- if ``input`` is a collection of keys, a 2-dimensional pandas DataFrame or numpy array is
  passed to the transformer

With respect to the discussion about robustness above, ballet tries to pass the most obvious
objects to the transformer. For example, if the raw data is a pandas ``DataFrame`` and ``input``
is a scalar key, ballet tries to pass a ``Series`` to the transformer. If that fails in a
predictable way (i.e. the transformer appears to not be able to handle that data type), then ballet
tries again with the next most obvious input data type (a 1-d numpy array), continuous on to a
pandas ``DataFrame`` with one column and finally a 2-d numpy array with one column. The same
principles apply when ``input`` is a collection of keys, except ballet will not try to pass any 1-d
data.

Transformers
^^^^^^^^^^^^

The ``transformer`` field accepts either a transformer-like object or a list of transformer-like
objects. By *transformer-like*, we mean objects that satisfy the scikit-learn Transformer API,
having ``fit``, ``transform``, and ``fit_transform`` implementations.

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

Many feature engineering primitives are also available in scikit-learn.

Preprocessing
^^^^^^^^^^^^^

See `sklearn.preprocessing`_ for a collection of useful preprocessing transformers.

Operating on groups
^^^^^^^^^^^^^^^^^^^

See :py:class:`ballet.eng.base.GroupedFunctionTransformer` and
:py:class:`ballet.eng.base.GroupwiseTransformer`.

Addressing missing values
^^^^^^^^^^^^^^^^^^^^^^^^^

See `sklearn.impute`_ and :py:mod:`ballet.eng.missing`.

Operating on time series data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

See :py:mod:`ballet.eng.ts`.

Other primitives
^^^^^^^^^^^^^^^^

See :py:class:`ballet.eng.base.SimpleFunctionTransformer` and
:py:class:`ballet.eng.base.ConditionalTransformer`.

Rolling your own transformers
-----------------------------

As you come up with more creative features, you may find that you need to create your own
transformer classes. Here are some tips for creating your own transformers.

1. Build off of :py:class:`ballet.eng.base.BaseTransformer` which inherits from
   :py:class:`sklearn.base.BaseEstimator`, :py:class:`sklearn.base.TransformerMixin`, and
   :py:class:`ballet.eng.base.NoFitMixin`.
2. Read the `scikit-learn documentation on a similar topic <https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator>`_.
   (Note that this documentation page is likely overkill for the types of transformers you may be
   implemeting.

Example
^^^^^^^

Let's create a feature that captures whether a string variable is the longest value observed in the
data. This is a silly feature for a number of reasons, so don't take it too seriously, but it
demonstrates the steps required to roll your own transformer.

.. code-block:: python

   from ballet import Feature
   from ballet.eng.base import BaseTransformer

   input = 'Exterior 1st'

   class LongestStringValue(BaseTransformer):

       def fit(self, X, y=None):
           self.longest_string_length_ = X.str.len().max()
           return self

       def transform(self, X):
           return X.str.len() >= self.longest_string_length_

    transformer = LongestStringValue()
    feature = Feature(input=input, transformer=transformer)

Okay, let's unpack what happened here. First, we declared the input to this feature, ``'Exterior
1st'``, a scalar key, so the feature will receive a pandas ``Series`` as the input ``X``. Next we
created a new class that inherits from ``BaseTransformer``. The transformer does not have any
"hyperparameters" so we can skip defining an ``__init__`` method. Following the scikit-learn
conventions, any learning from training data is done in the fit stage, and any learned parameters
are set on the class instance with names suffixed by a single underscore. The fit method should
also return ``self`` so that the ``fit_transform`` method defined on ``BaseTransformer`` can work.
We were able to assume that ``X`` is a series, and thus has the ``.str`` vectorized string
accessor. (If this were to be a new feature engineering primitive that would be used in more than
this one situation, we might want to add logic to allow the feature to operate on a DataFrame as
well.) Next, in the transform stage, we check for each new instance whether the length is greater
than or equal to the longest string length observed in the training data. The result will be a 1-d
arrray (series) of bools. Finally, having created the transformer class, we create an instance of
it and create our Feature object.

Further reading
---------------

- :py:class:`ballet.feature.Feature`
- :py:class:`ballet.pipeline.FeatureEngineeringPipeline`

.. _`sklearn.preprocessing`: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
.. _`sklearn.impute`: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.impute
