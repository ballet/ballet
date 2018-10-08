#!/usr/bin/env python

from {{ cookiecutter.project_slug }}.conf import here, get
from {{ cookiecutter.project_slug }}.load_data import load_data
from {{ cookiecutter.project_slug }}.features.build_features import build_features, get_contrib_features

from ballet.validation import validate

project = {
    'get': get,
    'get_contrib_features': get_contrib_features,
    'build_features': build_features,
    'here': here,
}

validate(project)
