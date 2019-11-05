from setuptools import setup, find_packages

requirements = [
    'ballet>=0.5.3',
    'Click>=6.0',
]

setup(
    author='{{ cookiecutter.full_name.replace("\'", "\\\'") }}',
    author_email='{{ cookiecutter.email }}',
    entry_points={
        'console_scripts': ['{{ cookiecutter.project_slug }}-engineer-features={{ cookiecutter.project_slug }}.features:main'],
    },
    install_requires=requirements,
    name='{{ cookiecutter.project_name }}',
    packages=find_packages(include=['{{ cookiecutter.project_slug }}', '{{ cookiecutter.project_slug }}.*']),
    url='https://github.com/{{ cookiecutter.github_owner }}/{{ cookiecutter.project_slug }}',
)
