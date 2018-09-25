[![PyPI Shield](https://img.shields.io/pypi/v/ballet.svg)](https://pypi.python.org/pypi/ballet)
[![Travis CI Shield](https://travis-ci.org/HDI-Project/ballet.svg?branch=master)](https://travis-ci.org/HDI-Project/ballet)

# ballet

A **light**weight framework for collaborative data science projects with a focus on **feat**ure engineering.

*ballet* defines robust `Feature` and `FeaturePipeline` objects built
alongside the `sklearn_pandas` project, as well as providing a host of other functionality.

- Free software: MIT license
- Documentation: https://hdi-project.github.io/ballet
- Homepage: https://github.com/HDI-Project/ballet

*ballet.eng* is a library of powerful, versatile transformers for feature engineering built
on top of the base `Feature` abstraction.

The *scikit_learn* paradigm of transformers with `fit` and `transform` methods is
well-suited for feature engineering. However, whereas data scientists may be used to working
on *pandas* `DataFrame`s, the *scikit_learn* library works on top of *NumPy* and is
ill-suited for operating on `DataFrame`s. This package defines a host of transformers for
data preprocessing, time series feature engineering, and much more.

## Installation

``` shell
git clone git@github.com:HDI-Project/ballet
cd ballet
make install
```

## Basic Usage

``` python
import ballet
```
