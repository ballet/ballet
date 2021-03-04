import logging
from typing import Optional, Union

import ballet

TRACE = 7
SIMPLE_LOG_FORMAT = r'%(levelname)s - %(message)s'
DETAIL_LOG_FORMAT = r'[%(asctime)s] {%(name)s: %(filename)s:%(lineno)d} %(levelname)s - %(message)s'  # noqa E501

logging.addLevelName(TRACE, 'TRACE')
logger = logging.getLogger(ballet.__name__)
_handler = None


def enable(logger: Union[str, logging.Logger] = logger,
           level: Union[str, int] = logging.INFO,
           format: str = SIMPLE_LOG_FORMAT,
           echo: bool = True):
    """Enable simple console logging for this module

    Args:
        logger: logger to enable. Defaults to ballet logger.
        level: logging level, either as string (``'INFO'``) or as int
            (``logging.INFO`` or ``20``). Defaults to ``'INFO'``.
        format : logging format. Defaults to :py:const:`SIMPLE_LOG_FORMAT`.
        echo: Whether to log a message at the configured log level to
            confirm that logging is enable. Defaults to True.
    """
    if isinstance(logger, str):
        logger = logging.getLogger(logger)

    global _handler
    if _handler is None:
        _handler = logging.StreamHandler()
        _handler.setFormatter(logging.Formatter(format))

    levelInt: int = logging._checkLevel(level)  # type: ignore
    levelName: str = logging.getLevelName(levelInt)

    logger.setLevel(levelName)
    # _handler.setLevel(levelName)  # might defeat the point

    if _handler not in logger.handlers:
        logger.addHandler(_handler)

    if echo:
        logger.log(levelInt, f'Logging enabled at level {levelName}.')


class LevelFilter:
    """Logging filter for log records at an exact level

    Emits only log records that have the exact level that is given to the
    filter.

    Args:
        level: numeric logging level to filter

    Example usage:

        >>> debugFilter = LevelFilter(logging.DEBUG)
        >>> logger.addFilter(debugFilter)
    """

    def __init__(self, level: int):
        self.__level = level

    def filter(self, logRecord: logging.LogRecord) -> int:
        # Is the specified record to be logged? Returns zero for no, nonzero
        # for yes.
        return 1 if logRecord.levelno == self.__level else 0


class LoggingContext:
    """Logging context manager

    Configure the given logger to emit messages at and above the configured
    logging level and using the configured logging handler. Useful to
    temporarily set a lower (or higher) log level or to temporarily add a
    local handler. After the context exits, the original state of the logger
    will be restored.

    If a handler is not provided, messages will continue to be logged to the
    existing handlers attached to the logger. If the logger's level is
    lowered, the handlers may

    Args:
        logger: logger
        level: string or numeric logging level
        handler: log handler if not is already configured
        close: whether to close the handler after context exits.
            Defaults to True.

    Example usage:

        >>> with LoggingContext(logger, level='DEBUG'):
        ...     logger.debug('some message')

    Source: <https://docs.python.org/3/howto/logging-cookbook.html#using-a-context-manager-for-selective-logging>
    """  # noqa E501

    def __init__(self,
                 logger: logging.Logger,
                 level: Union[str, int, None] = None,
                 handler: Optional[logging.Handler] = None,
                 close: bool = True):
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
