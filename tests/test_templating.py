import unittest
from unittest.mock import patch

from ballet.templating import render_project_template


class TemplatingTest(unittest.TestCase):

    @patch('ballet.templating.cookiecutter')
    def test_render_project_template(self, mock_cookiecutter):
        render_project_template()

        args, _ = mock_cookiecutter.call_args

        self.assertEqual(len(args), 1)
        path = args[0]
        self.assertIn('project_template', str(path))
