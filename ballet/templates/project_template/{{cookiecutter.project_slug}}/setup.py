from setuptools import setup, find_packages

requirements = [
    'ballet==0.13.1',
]

setup(
    name='{{ cookiecutter.package_slug }}',
    version='0.1.0-dev',
    packages=find_packages(where='src', include=('{{ cookiecutter.package_slug }}', '{{ cookiecutter.package_slug }}.*')),
    package_dir={'': 'src'},
    install_requires=requirements,

    # metadata
    author='{{ cookiecutter.full_name.replace("\'", "\\\'") }}',
    author_email='{{ cookiecutter.email }}',
    description='A data science project built on the Ballet framework',
    url='https://github.com/{{ cookiecutter.github_owner }}/{{ cookiecutter.project_slug }}',
)
