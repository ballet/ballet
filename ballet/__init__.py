# -*- coding: utf-8 -*-

"""Top-level package for ballet."""

__author__ = """Micah Smith"""
__email__ = 'micahs@mit.edu'
__version__ = '0.5.1-dev'


# re-export some names
from ballet.feature import *  # noqa
from ballet.contrib import *  # noqa


# configure module-level logging
import logging  # noqa E402
from ballet.util.log import logger  # noqa E402
logger.addHandler(logging.NullHandler())

# filter warnings
import warnings  # noqa E402
warnings.filterwarnings(
    action="ignore", module="scipy", message="^internal gelsd")
