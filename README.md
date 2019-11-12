[![PyPI Shield](https://img.shields.io/pypi/v/ballet.svg)](https://pypi.org/project/ballet)
[![Travis CI Shield](https://travis-ci.org/HDI-Project/ballet.svg?branch=master)](https://travis-ci.org/HDI-Project/ballet)
[![codecov Shield](https://codecov.io/gh/HDI-Project/ballet/branch/master/graph/badge.svg)](https://codecov.io/gh/HDI-Project/ballet)


# ballet

A **light**weight framework for collaborative data science projects through **feat**ure
engineering.

*Ballet* is under active development, please [report all
bugs](https://hdi-project.github.io/ballet/contributing.html#report-bugs).

- Free software: MIT license
- Documentation: https://hdi-project.github.io/ballet
- Homepage: https://github.com/HDI-Project/ballet

## Overview

Ballet projects maintain a *feature engineering pipeline invariant*: at any point, the code
and features within a project repository can be used for end-to-end feature engineering for
a given dataset. To expand on an existing feature engineering pipeline, well-structured
feature source code submissions can be proposed by contributors and extensively validated
for compatibility and performance.

How do you use the Ballet framework? First, you render a brand new ballet project from a
provided project template using a quickstart command and push it to GitHub. This project
contains an "empty" feature engineering pipeline. Next, you and your collaborators write
feature engineering source code and submit pull requests to include your new features in the
project and grow the pipeline. Features are instances of `ballet.Feature`, usually
leveraging `ballet.eng`, a library of versatile transformers and transformer building blocks
for developing features that learn. Once new pull requests are received by your project, a
continuous integration service runs a streaming logical feature selection algorithm. This is
part of an extensive feature validation suite that makes sure both that the proposed
features are useful and that they can be safely integrated into your project. If the
proposed feature is accepted, it can be safely merged.

<img src="./docs/_static/feature_lifecycle.png" alt="Ballet Feature Lifecycle" width="500" />
