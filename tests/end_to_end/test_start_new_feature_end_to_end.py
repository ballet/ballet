from unittest.mock import patch

from ballet.templating import start_new_feature
from ballet.util import work_in


def test_start_new_feature(quickstart, caplog):
    with work_in(quickstart.project.path):
        with patch('ballet.project.Project.from_cwd') as mock_from_cwd:
            mock_from_cwd.return_value = quickstart.project
            extra_context = {
                'username': 'username',
                'featurename': 'featurename',
            }
            start_new_feature(no_input=True, extra_context=extra_context)

    assert 'Start new feature successful' in caplog.text
