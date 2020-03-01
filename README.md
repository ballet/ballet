[![PyPI Shield](https://img.shields.io/pypi/v/ballet.svg)](https://pypi.org/project/ballet)
[![Travis CI Shield](https://travis-ci.com/HDI-Project/ballet.svg?branch=master)](https://travis-ci.com/HDI-Project/ballet)
[![codecov Shield](https://codecov.io/gh/HDI-Project/ballet/branch/master/graph/badge.svg)](https://codecov.io/gh/HDI-Project/ballet)


# ballet

A **light**weight framework for collaborative, open-source data science 
projects through **feat**ure engineering.

- Free software: MIT license
- Documentation: https://hdi-project.github.io/ballet
- Homepage: https://github.com/HDI-Project/ballet

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
data science process into modular patches - standalone units of contribution -
that can then be intelligently combined, representing objects like "feature", 
"labeling function", or "prediction task definition". Collaborators work in
parallel to write patches and submit them to a repo. Our software framework
provides the underlying functionality to merge high-quality contributions,
collect modules from the file system, and compose the accepted contributions
into a single product. It also provides a familiar notebook-based development
experience that is friendly to data scientists and other inexperienced
open-source contributors. We don't require any computing infrastructure beyond
that which is commonly used in open-source software development.

Currently, Ballet focuses on supporting collaboratively developing 
*feature engineering pipelines*, an important part of many data science
projects. Individual features are represented as separate Python modules,
declaring the subset of a dataframe that they operate on and a
scikit-learn-style learned transformer that extracts feature values from the raw
data. Ballet collects individual features and composes them into a feature
engineering pipeline. At any point, a project built on Ballet can be installed
for end-to-end feature engineering on new data instances for the same problem.
How do we ensure the feature engineering pipeline is always useful? Ballet 
thoroughly validates proposed features for correctness and machine learning 
performance, using an extensive test suite and a novel streaming logical 
feature selection algorithm. Accepted features can be automatically merged by
the ballet GitHub app into projects.

<img src="./docs/_static/feature_lifecycle.png" alt="Ballet Feature Lifecycle" width="400" />

## Next steps

*Are you a data owner or project maintainer that wants to organize a
collaboration?*

👉 Check out the [Ballet Maintainer Guide](https://hdi-project.github.io/ballet/maintainer_guide.html)

*Are you a data scientist or enthusiast that wants to join a collaboration?*

👉 Check out the [Ballet Contributor Guide](https://hdi-project.github.io/ballet/contributor_guide.html)

*Want to learn about how Ballet enables Better Feature Engineering ™️?*

👉 Check out the [Feature Engineering Guide](https://hdi-project.github.io/ballet/feature_engineering_guide.html)

*Want to see a demo collaboration in progress and maybe even participant yourself?*

👉 Check out the [ballet-predict-house-prices](https://github.com/HDI-Project/ballet-predict-house-prices) project
