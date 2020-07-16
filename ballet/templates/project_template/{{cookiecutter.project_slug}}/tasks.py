from invoke import task


@task
def install(c, system=False):
    """install the package to the active Python's site-packages"""
    c.run('pip install -r requirements.txt')


def rmdir(c, name):
    c.run('rm -rf {name}')


def rm(c, pattern):
    c.run(f'find . -name \'{pattern}\' -exec rm -rf {{}} +')


@task
def clean(c):
    """Clean python artifacts"""
    for name in ['build', 'dist', '.eggs']:
        rmdir(c, name)

    for pattern in ['*.egg-info', '*.egg', '*.pyc', '*.pyo', '*~',
                    '__pycache__']:
        rm(c, pattern)
