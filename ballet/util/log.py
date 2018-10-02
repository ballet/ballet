import logging

import ballet


LOG_FORMAT = r'[%(asctime)s] {%(name)s: %(filename)s:%(lineno)d} %(levelname)s - %(message)s'  # noqa E501
SIMPLE_LOG_FORMAT = r'%(asctime)s %(levelname)s - %(message)s'

logger = logging.getLogger(ballet.__name__)
_handler = None


def enable(logger=logger, level=logging.INFO):
    """Enable simple console logging for this module"""
    global _handler
    if _handler is None:
        _handler = logging.StreamHandler()
        formatter = logging.Formatter(LOG_FORMAT)
        _handler.setFormatter(formatter)

    logger.setLevel(level)
    _handler.setLevel(level)

    if _handler not in logger.handlers:
        logger.addHandler(_handler)

    levelName=logging._levelToName[level]
    logger.log(
        level, 'Logging enabled at level {name}.'.format(name=levelName))
