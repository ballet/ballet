from ballet.validation.common import load_spec
from ballet.validation.feature_acceptance.validator import (
    AlwaysAccepter, RandomAccepter,)
from ballet.validation.project_structure.validator import (
    ProjectStructureValidator,)


def test_load_spec_from_name(caplog):
    spec = 'ballet.validation.project_structure.validator.ProjectStructureValidator'  # noqa
    expected_class = ProjectStructureValidator
    cls, params = load_spec(spec)
    assert cls is expected_class
    assert isinstance(params, dict)


def test_load_spec_noparams():
    spec = {
        'name': 'ballet.validation.feature_acceptance.validator.AlwaysAccepter',  # noqa
    }
    expected_class = AlwaysAccepter
    cls, params = load_spec(spec)
    assert cls is expected_class


def test_load_spec_params():
    threshold = 0.88
    spec = {
        'name': 'ballet.validation.feature_acceptance.validator.RandomAccepter',  # noqa
        'params': {
            'threshold': threshold
        }
    }
    expected_class = RandomAccepter
    cls, params = load_spec(spec)
    assert cls is expected_class
    assert params['threshold'] == threshold
