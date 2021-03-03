import logging
from unittest.mock import MagicMock, patch

import pytest

import ballet.util.log
from ballet.exc import SkippedValidationTest
from ballet.validation.main import (  # noqa F401
    _check_project_structure, _evaluate_feature_performance,
    _load_validator_class_params, _prune_existing_features,
    _validate_feature_api, validate, validation_stage,)


def test_validation_stage_success(caplog):
    caplog.set_level(logging.DEBUG, logger=ballet.util.log.logger.name)

    @validation_stage('do something')
    def call():
        return True

    call()

    assert 'DONE' in caplog.text


def test_validation_stage_skipped(caplog):
    caplog.set_level(logging.DEBUG, logger=ballet.util.log.logger.name)

    @validation_stage('do something')
    def call():
        raise SkippedValidationTest

    call()

    assert 'SKIPPED' in caplog.text


def test_validation_stage_failure():
    exc = ValueError

    @validation_stage('do something')
    def call():
        raise exc

    with pytest.raises(exc):
        call()


@patch('ballet.validation.project_structure.validator.ChangeCollector')
@patch('ballet.validation.main.import_module_from_modname')
def test_load_validator_class_params_noparams(mock_import, _, caplog):
    """Check that _load_validator_class_params works"""

    import ballet.validation.project_structure.validator
    from ballet.validation.project_structure.validator import (
        ProjectStructureValidator,)
    mock_import.return_value = \
        ballet.validation.project_structure.validator

    mock_project = MagicMock()
    modname = 'ballet.validation.project_structure.validator'
    clsname = 'ProjectStructureValidator'
    path = modname + '.' + clsname
    mock_project.config.get.return_value = path

    config_key = None

    with caplog.at_level(logging.DEBUG, logger=ballet.util.log.logger.name):
        make_validator = _load_validator_class_params(
            mock_project, config_key)
    assert clsname in caplog.text

    validator = make_validator(mock_project)
    assert isinstance(validator, ProjectStructureValidator)


@pytest.mark.xfail
def test_load_validator_class_params_withparams():
    """Check that _load_validator_class_params correctly applies kwargs"""
    raise NotImplementedError


@pytest.mark.xfail
def test_check_project_structure():
    raise NotImplementedError


@pytest.mark.xfail
def test_validate_feature_api():
    raise NotImplementedError


@pytest.mark.xfail
def test_evaluate_feature_performance():
    raise NotImplementedError


@pytest.mark.xfail
def test_prune_existing_features():
    raise NotImplementedError


@pytest.mark.xfail
def test_validate():
    raise NotImplementedError
