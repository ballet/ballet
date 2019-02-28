import unittest
from unittest.mock import PropertyMock, patch

from ballet.compat import pathlib
from ballet.exc import ConfigurationError
from ballet.project import (
    DEFAULT_CONFIG_NAME, Project, find_configs, get_config_paths,
    make_config_get)


class ProjectTest(unittest.TestCase):

    def test_get_config_paths(self):
        package_root = "."
        config_paths = get_config_paths(package_root)

        self.assertGreater(len(config_paths), 0)

        for path in config_paths:
            self.assertIn(DEFAULT_CONFIG_NAME, str(path))

    @patch('ballet.project.get_config_paths')
    def test_find_configs_no_paths_fails(self, mock_get_config_paths):
        mock_get_config_paths.return_value = []
        package_root = None
        with self.assertRaises(ConfigurationError):
            find_configs(package_root)

    @patch('ballet.project.get_config_paths')
    def test_find_configs_no_valid_paths_fails(self, mock_get_config_paths):
        mock_get_config_paths.return_value = [
            pathlib.Path('/foo', 'bar', 'baz', DEFAULT_CONFIG_NAME)
        ]
        package_root = None
        with self.assertRaises(ConfigurationError):
            find_configs(package_root)

    @patch.object(pathlib, 'Path')
    @patch('ballet.project.find_configs')
    def test_config_get(self, mock_find_configs, mock_Path):
        config1 = {
            'problem': {
                'name': 'foo',
                'kind': 'Z',
            },
        }
        path1 = None
        config2 = {
            'problem': {
                'name': 'bar',
                'type': 'A',
            },
        }
        path2 = None
        config_info = [(config1, path1), (config2, path2)]
        mock_find_configs.return_value = config_info

        package_root = None
        get = make_config_get(package_root)

        self.assertEqual(get('problem', 'name'), 'foo')
        self.assertEqual(get('problem', 'kind'), 'Z')
        self.assertEqual(get('problem', 'type'), 'A')

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
