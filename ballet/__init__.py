# -*- coding: utf-8 -*-

"""Top-level package for ballet."""

__author__ = 'Micah Smith'
__email__ = 'micahs@mit.edu'
__version__ = '0.13.1'

# filter warnings
import warnings  # noqa E402
warnings.filterwarnings(
    action='ignore', module='scipy', message='^internal gelsd')

# silence sklearn deprecation warnings
import logging  # noqa E402
logging.captureWarnings(True)
import sklearn  # noqa
logging.captureWarnings(False)
warnings.filterwarnings(
    action='ignore', module='sklearn', category=DeprecationWarning)
warnings.filterwarnings(
    action='ignore', module='sklearn', category=FutureWarning)

# configure module-level logging
from ballet.util.log import logger  # noqa E402
logger.addHandler(logging.NullHandler())

# re-export some names
from ballet.client import b  # noqa
from ballet.contrib import *  # noqa
from ballet.feature import *  # noqa
