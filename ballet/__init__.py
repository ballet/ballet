# -*- coding: utf-8 -*-

"""Top-level package for ballet."""

__author__ = """Micah Smith"""
__email__ = 'micahs@mit.edu'
__version__ = '0.4.1'

# pathlib compatibility
import sys
if sys.version_info < (3, 5, 0):
    import pathlib2 as pathlib
else:
    import pathlib  # noqa F401


# re-export some names
from ballet.feature import *  # noqa
from ballet.contrib import *  # noqa


# configure module-level logging
import logging  # noqa E402
from ballet.util.log import logger  # noqa E402
logger.addHandler(logging.NullHandler())
