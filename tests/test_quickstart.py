import unittest
from unittest.mock import patch

from ballet.quickstart import main


class QuickstartTest(unittest.TestCase):

    @patch('ballet.quickstart.cookiecutter')
    def test_quickstart(self, mock_cookiecutter):
        main()

        args, _ = mock_cookiecutter.call_args

        self.assertEqual(len(args), 1)
        path = args[0]
        self.assertIn('project_template', str(path))
