import pathlib
import unittest
from unittest.mock import patch

from ballet.templating import (
    _stringify_path, render_feature_template, render_project_template)


class TemplatingTest(unittest.TestCase):

    def test__stringify_path_pathlike(self):
        obj = pathlib.Path()
        output = _stringify_path(obj)
        self.assertIsInstance(output, str)

    def test__stringify_path_not_pathlike(self):
        obj = object()
        output = _stringify_path(obj)
        self.assertNotIsInstance(output, str)

    @patch('ballet.templating.cookiecutter')
    def test_render_project_template(self, mock_cookiecutter):
        render_project_template()

        args, _ = mock_cookiecutter.call_args

        self.assertEqual(len(args), 1)
        path = args[0]
        self.assertIn('project_template', str(path))

    @patch('ballet.templating.cookiecutter')
    def test_render_feature_template(self, mock_cookiecutter):
        render_feature_template()

        args, _ = mock_cookiecutter.call_args

        self.assertEqual(len(args), 1)
        path = args[0]
        self.assertIn('feature_template', str(path))
