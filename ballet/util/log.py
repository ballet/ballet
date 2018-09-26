import logging

import ballet

logger = logging.getLogger(ballet.__name__)

LOG_FORMAT = r'[%(asctime)s] {%(name)s: %(filename)s:%(lineno)d} %(levelname)s - %(message)s'  # noqa E501
SIMPLE_LOG_FORMAT = r'%(asctime)s %(levelname)s - %(message)s'


def enable(level=logging.INFO):
    """Enable simple console logging for this module"""
    logger.setLevel(logging.DEBUG)
    stderr_handler = logging.StreamHandler()
    stderr_handler.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)
    logger.info('Logging enabled.')
