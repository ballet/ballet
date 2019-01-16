[![PyPI Shield](https://img.shields.io/pypi/v/ballet.svg)](https://pypi.python.org/pypi/ballet)
[![Travis CI Shield](https://travis-ci.org/HDI-Project/ballet.svg?branch=master)](https://travis-ci.org/HDI-Project/ballet)
[![codecov Shield](https://codecov.io/gh/HDI-Project/ballet/branch/master/graph/badge.svg)](https://codecov.io/gh/HDI-Project/ballet)


# ballet

A **light**weight framework for collaborative data science projects through **feat**ure engineering.

Ballet projects maintain a *feature engineering pipeline invariant*: at any point, the code and features within a
project repository can be used for end-to-end feature engineering for a given dataset. To expand on an existing feature
engineering pipeline, well-structured feature source code submissions can be proposed by contributors and extensively
validated for compatibility and performance.

Ballet provides the following functionality:
- `ballet-quickstart`, a command to generate a new predictive modeling project that uses Ballet framework
- `Feature` objects, that store feature metadata as well as a robust `DelegatingRobustTransformer` transformer pipeline
    built alongside the `sklearn_pandas` project.
- `ballet.eng`, a library of versatile transformers and transformer building blocks for developing features that learn.
- an extensive feature validation suite, that checks project structure and feature API adherence and runs a streaming
    logical feature selection algorithm.

*Ballet* is under active development, please [report all
bugs](https://hdi-project.github.io/ballet/contributing.html#report-bugs).

- Free software: MIT license
- Documentation: https://hdi-project.github.io/ballet
- Homepage: https://github.com/HDI-Project/ballet
