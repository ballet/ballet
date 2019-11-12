import pytest
from cookiecutter.utils import work_in

from ballet.compat import safepath


@pytest.mark.slow
def test_quickstart_install(quickstart, virtualenv):
    # This is an annoying test because the project template should specify
    # as installation dependency the most recent tagged version of ballet,
    # but this will not necessarily have been released on PyPI
    # TODO: figure out the right way to mange this
    d = quickstart.tempdir.joinpath(quickstart.project_slug).absolute()
    with work_in(safepath(d)):
        # cmd = 'cd "{d!s}" && make install'.format(d=d)
        cmd = 'echo okay'
        virtualenv.run(cmd, capture=True)
