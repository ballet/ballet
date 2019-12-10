import logging

import ballet

SIMPLE_LOG_FORMAT = r'%(asctime)s %(levelname)s - %(message)s'
DETAIL_LOG_FORMAT = r'[%(asctime)s] {%(name)s: %(filename)s:%(lineno)d} %(levelname)s - %(message)s'  # noqa E501

logger = logging.getLogger(ballet.__name__)
_handler = None


def enable(logger=logger,
           level=logging.INFO,
           format=DETAIL_LOG_FORMAT,
           echo=True):
    """Enable simple console logging for this module"""
    global _handler
    if _handler is None:
        _handler = logging.StreamHandler()
        formatter = logging.Formatter(format)
        _handler.setFormatter(formatter)

    level = logging._checkLevel(level)
    levelName = logging._levelToName[level]

    logger.setLevel(level)
    _handler.setLevel(level)

    if _handler not in logger.handlers:
        logger.addHandler(_handler)

    if echo:
        logger.log(
            level, 'Logging enabled at level {name}.'.format(name=levelName))


class LevelFilter(object):
    def __init__(self, level):
        self.__level = level

    def filter(self, logRecord):
        return logRecord.levelno == self.__level


class LoggingContext(object):
    """Logging context manager

    Source: <https://docs.python.org/3/howto/logging-cookbook.html#using-a-context-manager-for-selective-logging>
    """  # noqa E501

    def __init__(self, logger, level=None, handler=None, close=True):
        self.logger = logger
        self.level = level
        self.handler = handler
        self.close = close

    def __enter__(self):
        if self.level is not None:
            self.old_level = self.logger.level
            self.logger.setLevel(self.level)
        if self.handler is not None:
            self.logger.addHandler(self.handler)

    def __exit__(self, et, ev, tb):
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        if self.handler is not None:
            self.logger.removeHandler(self.handler)
        if self.handler and self.close:
            self.handler.close()
        return None
