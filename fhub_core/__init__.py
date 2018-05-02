# -*- coding: utf-8 -*-

"""Top-level package for fhub_core."""

__author__ = """Micah Smith"""
__email__ = 'micahs@mit.edu'
__version__ = '0.3.5'

# pathlib compatibility
import sys
if sys.version_info < (3, 5, 0):
    import pathlib2 as pathlib
else:
    import pathlib


from fhub_core.feature import *  # noqa
from fhub_core.contrib import *  # noqa
