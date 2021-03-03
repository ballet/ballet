import re

import pytest
from pkg_resources import working_set


@pytest.mark.slow
def test_quickstart_install(quickstart, virtualenv):
    """Test that we can install the project resulting from quickstart

    Note that virtualenv.install_package is just bogus and should be avoided.
    """
    d = quickstart.tempdir.joinpath(quickstart.project_slug).resolve()

    # install ballet
    ballet_pkg = next(p for p in working_set if p.project_name == 'ballet')
    if ballet_pkg is None:
        raise RuntimeError(
            'something is not right; ballet should be installed when running'
            'tests!')

    virtualenv.run(f'python -m pip install -e {ballet_pkg.location}')

    # patch setup.py to unpin dependency
    with d.joinpath('setup.py').open('r') as f:
        setup_contents = f.read()
    setup_contents = re.sub(
        r'ballet==[0-9.+-]+',
        'ballet',
        setup_contents,
    )
    with d.joinpath('setup.py').open('w') as f:
        f.write(setup_contents)

    # install the quickstart project
    virtualenv.run(f'python -m pip install -e {d}')

    # should be installed now, or will fail with error code != 0
    virtualenv.run(
        f'python -c "import {quickstart.package_slug}"',
        capture=True)
