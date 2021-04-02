#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md', encoding='utf-8') as history_file:
    history = history_file.read()

requirements = [
    'black',
    'cookiecutter',
    'Click >= 6.0',
    'dataclasses; python_version == "3.6"',
    'dill',
    'dynaconf ~= 3.0',
    'funcy >= 1.14',
    'gitpython',
    'h5py',
    'numpy',
    'packaging',
    'pandas ~= 1.0',
    'pygithub ~= 1.54',
    'pyyaml',
    'requests',
    'scikit_learn >=0.20, <0.23; python_version == "3.6"',
    'scikit_learn >= 0.20; python_version > "3.6"',
    'scipy',
    'sklearn_pandas ~= 1.0',
    'stacklog',
]

extras = {
    'category_encoders': ['category_encoders >= 2.2.2'],
    'feature_engine': ['feature_engine ~= 1.0'],
    'featuretools': ['featuretools_sklearn_transformer >= 0.1'],
    'skits': ['skits >= 0.1.2'],
    'tsfresh': ['tsfresh >= 0.16'],
}
extras['all'] = [dep for deps in extras.values() for dep in deps]

test_requirements = [
    'coverage >= 4.5.1',
    'pytest >= 6',
    'pytest-cov >= 2.6',
    'pytest-virtualenv >= 1.7.0',
    'tox >= 2.9.1',
    'responses >= 0.13.2',
]

development_requirements = [
    # general
    'bump2version >= 1',
    'pip >= 9.0.1',
    'watchdog[watchmedo] >= 0.8.3',
    'invoke >= 1.4',
    'mypy >= 0.812',

    # docs
    'm2r2 >= 0.2.5',
    'sphinx ~= 3.0',
    'sphinx_rtd_theme >= 0.2.4',
    'sphinx-click >= 2.2',
    'sphinx-autodoc-typehints >= 1.11',
    'rstcheck',

    # style check
    'flake8 >= 3.5.0',
    'isort >= 5.0',

    # fix style issues
    'autopep8 >= 1.3.5',

    # distribute on PyPI
    'twine >= 1.10.0',
    'wheel >= 0.30.0',
]

setup(
    author="Micah Smith",
    author_email='micahs@mit.edu',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description='Core functionality for lightweight, collaborative data science projects',
    entry_points={
        'console_scripts': [
            'ballet=ballet.cli:cli',
        ],
    },
    extras_require={
        'test': test_requirements,
        'dev': development_requirements + test_requirements,
        **extras,
    },
    install_requires=requirements,
    license='MIT license',
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='ballet',
    name='ballet',
    packages=find_packages(include=['ballet', 'ballet.*']),
    python_requires='>=3.6.1',
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/ballet/ballet',
    version='0.13.1',
    zip_safe=False,
)
