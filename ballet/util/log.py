import logging

from funcy import wraps

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

    levelName = logging._levelToName[level]
    logger.log(
        level, 'Logging enabled at level {name}.'.format(name=levelName))


class LoggingContext(object):
    '''
    Logging context manager

    Source: <https://docs.python.org/3/howto/logging-cookbook.html #using-a-context-manager-for-selective-logging>  # noqa
    '''

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


class stacklog:
    """Stack log messages

    Args:
        method: log callable
        message: log message
        *args: args to log method
        **kwargs: kwargs to log method

    Example usage:

       with stacklog(logging.info, 'Running long function'):
           run_long_function()

       with stacklog(logging.info, 'Running error-prone function'):
           raise Exception

    This produces logging output:

        INFO:root:Running long function...
        INFO:root:Running long function...DONE
        INFO:root:Running error-prone function...
        INFO:root:Running error-prone function...FAILURE
    """

    def __init__(self, method, message, *args, **kwargs):
        self.method = method
        self.message = str(message)
        self.args = args
        self.kwargs = kwargs

    def _log(self, suffix=''):
        self.method(self.message + '...' + suffix, *self.args, **self.kwargs)

    _begin = _log

    def _succeed(self):
        self._log(suffix='DONE')

    def _fail(self):
        self._log(suffix='FAILURE')

    def __call__(self, func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper

    def __enter__(self):
        self._begin()
        return

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self._succeed()
        else:
            self._fail()
        return False
