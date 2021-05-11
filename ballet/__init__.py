# -*- coding: utf-8 -*-

"""Top-level package for ballet."""

__author__ = 'Micah Smith'
__email__ = 'micahs@mit.edu'
__version__ = '0.14.0'

# filter warnings
import warnings  # noqa E402
warnings.filterwarnings(
    action='ignore', module='scipy', message='^internal gelsd')

# silence sklearn deprecation warnings
import logging  # noqa E402
logging.captureWarnings(True)
import sklearn  # noqa E402
logging.captureWarnings(False)
warnings.filterwarnings(
    action='ignore', module='sklearn', category=DeprecationWarning)
warnings.filterwarnings(
    action='ignore', module='sklearn', category=FutureWarning)

# configure module-level logging
from ballet.util.log import logger  # noqa E402
logger.addHandler(logging.NullHandler())

# re-export some names
from ballet.client import b  # noqa E402
from ballet.contrib import collect_contrib_features  # noqa E402
from ballet.feature import Feature  # noqa E402
from ballet.project import load_config, Project  # noqa E402

# for feature development, you really only need these two members
__all__ = (
    'b',
    'Feature',
)
