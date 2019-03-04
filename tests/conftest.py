import tempfile

import pytest

from ballet.compat import pathlib


@pytest.fixture
def tempdir():
    """Tempdir fixture using tempfile.TemporaryDirectory"""
    with tempfile.TemporaryDirectory() as d:
        yield pathlib.Path(d)
