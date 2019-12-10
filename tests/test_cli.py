import unittest
from unittest.mock import ANY, patch

from click.testing import CliRunner

import ballet
from ballet.cli import (  # noqa F401
    cli, quickstart, start_new_feature, update_project_template, validate)


class CliTest(unittest.TestCase):

    def setUp(self):
        self.runner = CliRunner()

    def test_cli_version(self):
        result = self.runner.invoke(cli, '--version')
        self.assertEqual(result.exit_code, 0)
        self.assertIn(ballet.__version__, result.output)

    @patch('ballet.templating.render_project_template')
    def test_quickstart(self, mock_render):
        result = self.runner.invoke(quickstart)
        mock_render.assert_called_once_with()
        self.assertIn('Generating new ballet project', result.output)

    @patch('ballet.update.update_project_template')
    def test_update_project_template(self, mock_update):
        result = self.runner.invoke(update_project_template, ['--push'])
        mock_update.assert_called_once_with(
            push=True, project_template_path=None)
        self.assertIn('Updating project template', result.output)

    @patch('ballet.templating.start_new_feature')
    def test_start_new_feature(self, mock_start):
        result = self.runner.invoke(start_new_feature)
        mock_start.assert_called_once_with()
        self.assertIn('Starting new feature', result.output)

    @patch('ballet.project.Project.from_path')
    @patch('ballet.validation.main.validate')
    def test_validate_all(self, mock_validate, mock_from_path):
        result = self.runner.invoke(validate, ['-A'])
        mock_validate.assert_called_once_with(ANY, True, True, True, True)
        self.assertIn('Validating project', result.output)
