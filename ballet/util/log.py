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
    """Enable simple console logging for this module

    Args:
        logger (logging.Logger): logger to enable. Defaults to ballet logger.
        level (str, int): logging level, either as string (``"INFO"``) or as
            int (``logging.INFO`` or ``20``). Defaults to 'INFO'.
        format (str) : logging format. Defaults to
            ballet.util.log.DETAIL_LOG_FORMAT.
        echo (bool): Whether to log a message at the configured log level to
            confirm that logging is enable. Defaults to True.
    """
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
    """Logging filter for log records at an exact level

    Args:
        level (int): numeric logging level to filter

    Example usage:

        >>> debugFilter = LevelFilter(logging.DEBUG)
        >>> logger.addFilter(debugFilter)
    """

    def __init__(self, level):
        self.__level = level

    def filter(self, logRecord):
        return logRecord.levelno == self.__level


class LoggingContext(object):
    """Logging context manager
    
    Configure the given logger to emit messages at and above the configured 
    logging level and using the configured logging handler. Useful to 
    temporarily set a lower (or higher log level) or to temporarily add a 
    local handler. After the context exits, the original state of the logger
    will be restored.
    
    Args:
        logger (logging.Logger): logger
        level (Union[str,int]): string or numeric logging level
        handler (logging.Handler): log handler if not is already configured
        close (bool): whether to close the handler after context exits. 
            Defaults to True.
            
    Example usage:
    
        >>> with LoggingContext(logger, level='DEBUG'):
        ...     logger.debug('some message')
        
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
