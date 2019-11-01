import unittest
from unittest.mock import ANY, PropertyMock, patch

from ballet.project import Project, detect_github_username


class ProjectTest(unittest.TestCase):

    @unittest.expectedFailure
    def test_load_config_at_path(self):
        raise NotImplementedError

    @unittest.expectedFailure
    def test_load_config_in_dir(self):
        raise NotImplementedError

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
