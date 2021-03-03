import re

import pytest


@pytest.mark.slow
def test_quickstart_install(quickstart, virtualenv):
    """Test that we can install the project resulting from quickstart"""
    d = quickstart.tempdir.joinpath(quickstart.project_slug).resolve()

    # install development version of ballet
    # since ballet is present in the "working set", pytest-virtualenv will
    # run:
    #    cd <ballet source directory> && python setup.py -q develop
    # unfortunately this will install all ballet dependencies from the
    # network, though they should be cached if tests are being run
    virtualenv.install_package('ballet', build_egg=False)

    # patch setup.py to remove ballet dependency by removing the
    # install_requires key entirely
    with d.joinpath('setup.py').open('r') as f:
        setup_contents = f.read()
    setup_contents = re.sub(
        re.compile(r'^\s*install_requires=.+,$', flags=re.MULTILINE),
        '',
        setup_contents,
    )
    with d.joinpath('setup.py').open('w') as f:
        f.write(setup_contents)

    # install the current package
    # can't use virtualenv.install_package, as it appears to be broken
    virtualenv.run(f'python -m pip install -e {d}', capture=False)

    # should be installed now, or will fail with error code != 0
    virtualenv.run(
        f'python -c "import {quickstart.package_slug}"',
        capture=True)
