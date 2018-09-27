import logging

import ballet


LOG_FORMAT = r'[%(asctime)s] {%(name)s: %(filename)s:%(lineno)d} %(levelname)s - %(message)s'  # noqa E501
SIMPLE_LOG_FORMAT = r'%(asctime)s %(levelname)s - %(message)s'

logger = logging.getLogger(ballet.__name__)
_handler = None


def enable(level=logging.INFO):
    """Enable simple console logging for this module"""
    name=logging._levelToName[level]
    global _handler
    if _handler is None:
        logger.setLevel(level)
        _handler = logging.StreamHandler()
        _handler.setLevel(level)
        formatter = logging.Formatter(LOG_FORMAT)
        _handler.setFormatter(formatter)
        logger.addHandler(_handler)
        logger.log(level, 'Logging enabled at level {name}.'.format(name=name))
    else:
        logger.setLevel(level)
        _handler.setLevel(level)
        logger.log(level, 'Logging changed to level {name}.'.format(name=name))
