import unittest
from unittest.mock import PropertyMock, patch

from ballet.compat import pathlib
from ballet.exc import ConfigurationError
from ballet.project import (
    DEFAULT_CONFIG_NAME, Project, get_config_path,
    make_config_get)


class ProjectTest(unittest.TestCase):

    def test_get_config_path(self):
        package_root = "."
        path = get_config_path(package_root)

        self.assertIn(DEFAULT_CONFIG_NAME, str(path))

    @unittest.expectedFailure
    def test_load_config_at_path(self):
        raise NotImplementedError

    @unittest.expectedFailure
    def test_load_config_in_dir(self):
        raise NotImplementedError

    @unittest.expectedFailure
    def test_config_get(self):
        raise NotImplementedError

    @patch.object(pathlib, 'Path')
    @patch('ballet.project._get_project_root_from_conf_path')
    @patch('ballet.project.load_config_at_path')
    def test_make_config_get(self,
                             mock_load_config_at_path,
                             mock_get_project_root_from_conf_path,
                             mock_Path):
        config = {
            'problem': {
                'name': 'foo',
                'kind': 'Z',
            },
        }
        mock_load_config_at_path.return_value = config

        conf_path = None
        get = make_config_get(conf_path)

        self.assertEqual(get('problem', 'name'), 'foo')
        self.assertEqual(get('problem', 'kind'), 'Z')

        # with default
        self.assertEqual(get('nonexistent', 'path', default=3), 3)

    @patch('ballet.project.Project.repo', new_callable=PropertyMock)
    @patch('ballet.project.get_pr_num')
    def test_project_pr_num(self, mock_get_pr_num, mock_repo):
        expected = 3
        mock_get_pr_num.return_value = expected

        package = None
        project = Project(package)
        self.assertEqual(project.pr_num, expected)
