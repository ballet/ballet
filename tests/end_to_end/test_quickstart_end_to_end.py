from ballet.compat import safepath

from cookiecutter.utils import work_in

def test_quickstart_install(quickstart, virtualenv):
    d = quickstart.tempdir.joinpath(quickstart.project_slug).absolute()
    with work_in(safepath(d)):
        cmd = 'cd "{d!s}" && make install'.format(d=d)
        virtualenv.run(cmd, capture=True)
