from unittest.mock import patch

import pytest

from ballet.templating import start_new_feature
from ballet.util import work_in


@pytest.mark.parametrize(
    'branching',
    [True, False],
)
def test_start_new_feature(quickstart, caplog, branching):
    default_branch = quickstart.project.branch
    with work_in(quickstart.project.path):
        with patch('ballet.project.Project.from_cwd') as mock_from_cwd:
            mock_from_cwd.return_value = quickstart.project
            extra_context = {
                'username': 'username',
                'featurename': 'featurename',
            }
            start_new_feature(
                branching=branching, no_input=True, extra_context=extra_context
            )

    assert 'Start new feature successful' in caplog.text

    if branching:
        assert quickstart.project.branch == 'username/feature-featurename'
    else:
        assert quickstart.project.branch == default_branch
