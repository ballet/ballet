#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

requirements = [
    'black',
    'cookiecutter',
    'Click>=6.0',
    'dill',
    'dynaconf<3',
    'funcy>=1.14',
    'gitpython',
    'h5py',
    'numpy',
    'packaging',
    'pandas',
    'pyyaml',
    'requests',
    'scikit_learn>=0.20',
    'scipy',
    'sklearn_pandas',
    'stacklog',
]

test_requirements = [
    'coverage>=4.5.1',
    'pytest>=3.4.2',
    'pytest-cov>=2.6',
    'pytest-virtualenv>=1.7.0',
    'tox>=2.9.1',
]

development_requirements = [
    # general
    'bump2version>=1',
    'pip>=9.0.1',
    'watchdog>=0.8.3',
    'invoke>=1.4',
    'mypy',

    # docs
    'm2r>=0.2.1',
    'Sphinx>=1.7.1,<3',  # todo - bug with m2r (can use m2r2); not supported by sphinx-click
    'sphinx_rtd_theme>=0.2.4',
    'sphinx-click>=1.4.1',
    'rstcheck',

    # style check
    'flake8>=3.5.0',
    'isort>=4.3.4,<=4.3.9',

    # fix style issues
    'autopep8>=1.3.5',

    # distribute on PyPI
    'twine>=1.10.0',
    'wheel>=0.30.0',
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
    },
    install_requires=requirements,
    license='MIT license',
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='ballet',
    name='ballet',
    packages=find_packages(include=['ballet', 'ballet.*']),
    python_requires='>=3.6',
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/HDI-Project/ballet',
    version='0.7.3',
    zip_safe=False,
)
