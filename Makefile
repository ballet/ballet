.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

.PHONY: help
help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

.PHONY: clean
clean: clean-build clean-pyc clean-test clean-docs ## remove all build, test, coverage, docs and Python artifacts

.PHONY: clean-build
clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

.PHONY: clean-pyc
clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

.PHONY: clean-test
clean-test: ## remove test and coverage artifacts
	rm -fr .tox
	rm -f .coverage
	rm -fr htmlcov
	rm -fr .pytest_cache
	rm -fr .mypy_cache

.PHONY: lint
lint: ## check style with flake8 and isort
	flake8 ballet tests
	isort -c --recursive ballet tests
	mypy ballet tests

.PHONY: fix-lint
fix-lint: ## fix lint issues using autopep8 and isort
	autopep8 --in-place --recursive --aggressive --aggressive ballet tests
	isort --apply --atomic --recursive ballet tests

.PHONY: test
test: ## run tests quickly with the default Python
	python -m pytest --basetemp=${ENVTMPDIR} --cov=ballet

.PHONY: test-fast
test-fast:  ## run tests that are not marked as 'slow'
	python -m pytest -m 'not slow'


.PHONY: test-all
test-all: ## run tests on every Python version with tox
	tox -r

.PHONY: coverage
coverage: ## check code coverage quickly with the default Python
	coverage run --source ballet -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

.PHONY: clean-docs
clean-docs: ## remove previously built docs
	rm -f docs/api/*.rst
	-$(MAKE) -C docs clean 2>/dev/null  # this fails if sphinx is not yet installed

.PHONY: docs
docs: clean-docs ## generate Sphinx HTML documentation, including API docs
	sphinx-apidoc --module-first --separate -o docs/api/ ballet
	$(MAKE) -C docs html

.PHONY: docs
check-docs: clean-docs ## check generation of Sphinx HTML documentation
	find . -name '*.rst' -exec rstcheck {} +
	sphinx-apidoc --module-first --separate -o docs/api/ ballet
	$(MAKE) -C docs linkcheck text

.PHONY: view-docs
view-docs: ## view current docs in browser
	$(BROWSER) docs/_build/html/index.html

.PHONY: serve-docs
serve-docs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst;*.md' -c '$(MAKE) -C docs html' -R -D .

.PHONY: release
release: dist ## package and upload a release
	twine upload dist/*

.PHONY: test-release
test-release: dist ## package and upload a release on TestPyPI
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

.PHONY: dist
dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

.PHONY: install
install: clean-build clean-pyc ## install the package to the active Python's site-packages
	pip install .

.PHONY: install-test
install-test: clean-build clean-pyc ## install the package and test dependencies
	pip  install .[test]

.PHONY: install-develop
install-develop: clean-build clean-pyc ## install the package in editable mode and dependencies for development
	pip install -e .[dev]

.PHONY: install-develop-all
install-develop-all: clean-build clean-pyc ## install the package in editable mode and dependencies for development
	pip install -e .[dev,all]
