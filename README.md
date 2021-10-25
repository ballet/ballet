[![PyPI Shield](https://img.shields.io/pypi/v/ballet.svg)](https://pypi.org/project/ballet)
[![Tests](https://github.com/ballet/ballet/workflows/Tests/badge.svg)](https://github.com/ballet/ballet/actions?query=workflow%3A%22Tests%22)
[![codecov Shield](https://codecov.io/gh/ballet/ballet/branch/master/graph/badge.svg)](https://codecov.io/gh/ballet/ballet)


# ballet

A **light**weight framework for collaborative, open-source data science
projects through **feat**ure engineering.

- Free software: MIT license
- Documentation: https://ballet.github.io/ballet
- Repo: https://github.com/ballet/ballet
- Project homepage: https://ballet.github.io

## Overview

While the open-source model for software development has led to successful, large-scale collaborations in building software applications, chess engines, and scientific analyses, data science has not benefited from this development paradigm. In part, this is due to the divide between the development processes used by software engineers and those used by data scientists.

Ballet tries to address this disparity. It is a lightweight software framework that supports collaborative data science development by composing a data science pipeline from a collection of modular patches that can be written in parallel. Ballet provides the underlying functionality to support interactive development, test and merge high-quality contributions, and compose the accepted contributions into a single product.

We have deployed Ballet for feature engineering collaborations on tabular survey datasets of public interest. For example, [predict-census-income](https://github.com/ballet/predict-census-income) is a large real-world collaborative project to engineer features from raw individual survey responses to the U.S. Census American Community Survey (ACS) and predict personal income. The resulting project is one of the largest data science collaborations GitHub, and outperforms state-of-the-art tabular AutoML systems and independent data science experts.

### The Ballet framework

Ballet includes several different pieces for enabling collaborative data science.

* The Ballet framework core is developed in this repository and includes:
    * the *feature definition* abstraction, a tuple of input variables and transformer steps (`ballet.feature`)
    * the *feature engineering pipeline* abstraction, a data flow graph over feature functions (`ballet.pipeline`)
    * the *transformer step* abstraction and a library of transformer steps that can be used in feature engineering (`ballet.tranformer`, `ballet.eng`)
    * a comprehensive feature validation library, that includes test suites and statistical methods for validating the machine learning performance and software quality of proposed feature definitions (`ballet.validation`)
    * functionality for programmatically collecting submitted feature definitions from file systems (`ballet.contrib`)
    * a project template  for individual Ballet projects that can be automatically updated with upstream template improvements (`ballet/templates/project_template`, `ballet.update`)
    * a command line tool for maintaining and developing Ballet projects (`ballet.cli`)
    * an interface to interact with Ballet projects following the project template (`ballet.project`)
    * an interactive client for users during development (`ballet.client`)
* [Assemblé](https://github.com/ballet/ballet-assemble): A development environment for Ballet collaborations on top of Jupyter Lab
* [Ballet Bot](https://github.com/ballet/ballet-bot): A bot to help manage Ballet projects on GitHub


## Next steps

### Learn more about Ballet

*Are you a data owner or project maintainer that wants to organize a
collaboration?*

👉 Check out the [Ballet Maintainer Guide](https://ballet.github.io/ballet/maintainer_guide.html)

*Are you a data scientist or enthusiast that wants to join a collaboration?*

👉 Check out the [Ballet Contributor Guide](https://ballet.github.io/ballet/contributor_guide.html)

*Do you want to learn about how Ballet enables Better Feature Engineering™️?*

👉 Check out the [Feature Engineering Guide](https://ballet.github.io/ballet/feature_engineering_guide.html)

You can also read our research paper about the Ballet framework and our case study analysis, which appeared at ACM CSCW 2021:

👉 [Enabling Collaborative Data Science Development with the Ballet Framework](https://dl.acm.org/doi/10.1145/3479575)

### Join a Ballet collaboration

The Ballet GitHub organization hosts several ongoing Ballet collaborations:

* [ballet/predict-house-prices](https://github.com/ballet/predict-house-prices): This is a sandbox collaboration that showcases Ballet. All submissions that pass the feature API validation will be automatically accepted.
* [ballet/predict-census-income](https://github.com/ballet/predict-census-income): This is a collaboration as part of a past research case study to better understand collaborative data science in action.
* [ballet/predict-life-outcomes](https://github.com/ballet/predict-life-outcomes): This is an ongoing collaboration to predict life outcomes for disadvantaged children and their families, inspired by the recent [Fragile Families Challenge](https://www.fragilefamilieschallenge.org/).

## Citing Ballet

If you use Ballet in your work, please consider citing the following paper:

```bibtex
@article{smith2021enabling,
    author = {Smith, Micah J. and Cito, J{\"u}rgen and Lu, Kelvin and Veeramachaneni, Kalyan},
    title = "Enabling Collaborative Data Science Development with the {Ballet} Framework",
    year = "2021",
    month = "October",
    volume = "5",
    pages = "1--39",
    doi = "10.1145/3479575",
    journal = "Proceedings of the {ACM} on Human-Computer Interaction",
    publisher = "{ACM}",
    language = "en",
    number = "CSCW2"
}
```
