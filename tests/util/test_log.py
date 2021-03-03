import logging
import random

import pytest

from ballet.util.log import LevelFilter, LoggingContext, enable


@pytest.fixture
def logger():
    name = str(random.randint(0, 1 << 10))
    return logging.getLogger(name)


@pytest.mark.parametrize('level', [logging.INFO, 'CRITICAL'])
def test_enable(logger, caplog, level):
    with caplog.at_level(level, logger=logger.name):
        enable(logger, level, echo=True)
    assert 'enabled' in caplog.text


def test_level_filter_matches(logger, caplog):
    enable(logger, level='DEBUG', echo=False)
    logger.addFilter(LevelFilter(logging.CRITICAL))

    # does log message at level CRITICAL
    with caplog.at_level(logging.CRITICAL, logger=logger.name):
        logger.critical('msg')

    assert caplog.text


def test_level_filter_not_matches(logger, caplog):
    enable(logger, level='DEBUG', echo=False)
    logger.addFilter(
        LevelFilter(logging.DEBUG))

    # does *not* log message at level INFO > DEBUG
    with caplog.at_level(logging.INFO, logger=logger.name):
        logger.info('msg')

    assert not caplog.text


def test_logging_context(logger, caplog):
    enable(logger, level='DEBUG', echo=False)
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        with LoggingContext(logger, level='INFO'):
            logger.debug('msg')
    assert not caplog.text
