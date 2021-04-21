[![PyPI Shield](https://img.shields.io/pypi/v/ballet.svg)](https://pypi.org/project/ballet)
[![Tests](https://github.com/ballet/ballet/workflows/Tests/badge.svg)](https://github.com/ballet/ballet/actions?query=workflow%3A%22Tests%22)
[![codecov Shield](https://codecov.io/gh/ballet/ballet/branch/master/graph/badge.svg)](https://codecov.io/gh/ballet/ballet)


# ballet

A **light**weight framework for collaborative, open-source data science
projects through **feat**ure engineering.

- Free software: MIT license
- Documentation: https://ballet.github.io/ballet
- Homepage: https://github.com/ballet/ballet

## Overview

Do you develop machine learning models? Do you work by yourself or on a team?
Do you share notebooks or are you committing code to a shared repository? In
contrast to successful, massively collaborative, open-source projects like
the Linux kernel, the Rails framework, Firefox, GNU, or Tensorflow, most
data science projects are developed by just a handful of people. But think if
the open-source community could leverage its ingenuity and determination to
collaboratively develop data science projects to predict the incidence of
disease in a population, to predict whether vulnerable children will be evicted
from their homes, or to predict whether learners will drop out of online
courses.

Our vision is to make collaborative data science possible by making it more
like open-source software development. Our approach is based on decomposing the
data science process into modular patches
that can then be intelligently combined, representing objects like "feature definition",
"labeling function", or "prediction task definition". Collaborators work in
parallel to write patches and submit them to a repo. The core Ballet framework
provides the underlying functionality to merge high-quality contributions,
collect modules from the file system, and compose the accepted contributions
into a single product. It also provides [Assemblé](https://github.com/ballet/ballet-assemble), a familiar notebook-based development
experience that is friendly to data scientists and other inexperienced
open-source contributors. We don't require any computing infrastructure beyond
that which is commonly used in open-source software development.

Currently, Ballet focuses on supporting collaboratively developing
*feature engineering pipelines*, an important part of many data science
projects. Individual feature definitions are represented as separate Python modules,
declaring the subset of a dataframe that they operate on and a
scikit-learn-style learned transformer that extracts feature values from the
raw data. Ballet collects individual feature definitions and composes them into a
feature engineering pipeline. At any point, a project built on Ballet can be
installed for end-to-end feature engineering on new data instances for the
same problem. How do we ensure the feature engineering pipeline is always
useful? Ballet thoroughly validates proposed feature definitions for correctness and
machine learning performance, using an extensive test suite and a novel
streaming feature definition selection algorithm. Accepted feature definitions can be
automatically merged by the [Ballet Bot](https://github.com/ballet/ballet-bot) into projects.

<img src="./docs/_static/feature_lifecycle.png" alt="Ballet Feature Lifecycle" width="400" />

## Next steps

*Are you a data owner or project maintainer that wants to organize a
collaboration?*

👉 Check out the [Ballet Maintainer Guide](https://ballet.github.io/ballet/maintainer_guide.html)

*Are you a data scientist or enthusiast that wants to join a collaboration?*

👉 Check out the [Ballet Contributor Guide](https://ballet.github.io/ballet/contributor_guide.html)

*Want to learn about how Ballet enables Better Feature Engineering™️?*

👉 Check out the [Feature Engineering Guide](https://ballet.github.io/ballet/feature_engineering_guide.html)

*Want to see a demo collaboration in progress and maybe even participate yourself?*

👉 Check out the [ballet-predict-house-prices](https://github.com/HDI-Project/ballet-predict-house-prices) project

## Source code organization

This is a quick overview to the Ballet core source code organization. For more information about contributing to Ballet core itself, see [here](https://ballet.github.io/ballet/contributing.html).

| path | description |
| ---- | ----------- |
| [`cli.py`](ballet/cli.py) | the `ballet` command line utility |
| [`client.py`](ballet/client.py) | the interactive client for users |
| [`contrib.py`](ballet/contrib.py) | collecting feature definitions from individual modules in source files in the file system |
| [`eng/base.py`](ballet/eng/base.py) | abstractions for transformers used in feature definitions, such as `BaseTransformer` |
| [`eng/{misc,missing,ts}.py`](ballet/eng/) | custom transformers for missing data, time series problems, and more |
| [`eng/external.py`](ballet/eng/external.py) | re-export of transformers from external libraries such as scikit-learn and feature_engine |
| [`feature.py`](ballet/feature.py) | the `Feature` abstraction |
| [`pipeline.py`](ballet/pipeline.py) | the `FeatureEngineeringPipeline` abstraction |
| [`project.py`](ballet/project.py) | the interface between a specific Ballet project and the core Ballet library, such as utilities to load project-specific information and the `Project` abstraction |
| [`templates/`](ballet/templates/) | cookiecutter templates for creating a new Ballet project or creating a new feature definition |
| [`templating.py`](ballet/templating.py) | user-facing functionality on top of the templates |
| [`transformer.py`](ballet/transformer.py) | wrappers for transformers that make them play nicely together in a pipeline |
| [`update.py`](ballet/update.py) | functionality to update the project template from a new upstream release |
| [`util/`](ballet/util/) | various utilities |
| [`validation/main.py`](ballet/validation/main.py) | entry point for all validation routines |
| [`validation/base.py`](ballet/validation/base.py) | abstractions used in validation such as the `FeaturePerformanceEvaluator` |
| [`validation/common.py`](ballet/validation/common.py) | common functionality used in validation, such as the ability to collect relevant changes between a current environment and a reference environment (such as a pull request vs a default branch) |
| [`validation/entropy.py`](ballet/validation/entropy.py) | statistical estimation routines used in feature definition selection algorithms, such as estimators for entropy, mutual information, and conditional mutual information |
| [`validation/feature_acceptance/`](ballet/validation/feature_acceptance/) | validation routines for feature acceptance
| [`validation/feature_pruning/`](ballet/validation/feature_pruning/) | validation routines for feature pruning |
| [`validation/feature_api/`](ballet/validation/feature_api/) | validation routines for feature APIs |
| [`validation/project_structure/`](ballet/validation/project_structure/) | validation routines for project structure |
