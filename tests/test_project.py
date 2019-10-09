import pathlib
import unittest
from unittest.mock import ANY, PropertyMock, patch

from ballet.project import (
    DEFAULT_CONFIG_NAME, Project, detect_github_username, get_config_path,
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
    def test_detect_github_username_config(self, mock_project_repo):
        expected_username = 'Foo Bar'

        # output of project.repo.config_reader().get_value(...)
        mock_get_value = (mock_project_repo
                          .return_value
                          .config_reader
                          .return_value
                          .get_value)
        mock_get_value.return_value = expected_username

        project = Project(None)
        username = detect_github_username(project)
        self.assertEqual(expected_username, username)

        mock_get_value.assert_called_with('github', 'user', default=ANY)

    @patch('ballet.project.Project.repo', new_callable=PropertyMock)
    @patch('ballet.project.get_pr_num')
    def test_project_pr_num(self, mock_get_pr_num, mock_repo):
        expected = 3
        mock_get_pr_num.return_value = expected

        package = None
        project = Project(package)
        self.assertEqual(project.pr_num, expected)
