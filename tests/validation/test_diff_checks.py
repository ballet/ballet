import pathlib
import unittest
from unittest.mock import Mock

from ballet.project import relative_to_contrib
from ballet.validation.project_structure.checks import (
    IsAdditionCheck, IsPythonSourceCheck, ModuleNameCheck,
    RelativeNameDepthCheck, SubpackageNameCheck, WithinContribCheck)

from .util import make_mock_project


class DiffCheckTest(unittest.TestCase):

    def setUp(self):
        self.contrib_module_path = 'foo/features/contrib'
        self.project = make_mock_project(
            None, None, '', self.contrib_module_path)

    def test_relative_to_contrib(self):
        diff = Mock(b_path='foo/features/contrib/abc.py')

        project = Mock()

        def mock_get(path):
            if path == 'contrib.module_path':
                return self.contrib_module_path
            else:
                raise KeyError
        project.config.get.side_effect = mock_get

        expected = pathlib.Path('abc.py')
        actual = relative_to_contrib(diff, project)

        self.assertEqual(actual, expected)

    def test_is_addition_check(self):
        checker = IsAdditionCheck(self.project)

        mock_diff = Mock(change_type='A')
        self.assertTrue(checker.do_check(mock_diff))

        mock_diff = Mock(change_type='B')
        self.assertFalse(checker.do_check(mock_diff))

    def test_is_python_source_check(self):
        checker = IsPythonSourceCheck(self.project)

        mock_diff = Mock(b_path='foo/features/contrib/user_bob/feature_1.py')
        self.assertTrue(checker.do_check(mock_diff))

        mock_diff = Mock(b_path='foo/features/contrib/user_bob/feature_1.xyz')
        self.assertFalse(checker.do_check(mock_diff))

    def test_within_contrib_check(self):
        checker = WithinContribCheck(self.project)

        mock_diff = Mock(b_path='foo/features/contrib/user_bob/feature_1.py')
        self.assertTrue(checker.do_check(mock_diff))

        mock_diff = Mock(b_path='foo/hack.py')
        self.assertFalse(checker.do_check(mock_diff))

    def test_subpackage_name_check(self):
        checker = SubpackageNameCheck(self.project)

        mock_diff = Mock(b_path='foo/features/contrib/user_bob/feature_1.py')
        self.assertTrue(checker.do_check(mock_diff))

        mock_diff = Mock(b_path='foo/features/contrib/bob/feature_1.py')
        self.assertFalse(checker.do_check(mock_diff))

    def test_feature_module_name_check(self):
        checker = ModuleNameCheck(self.project)

        mock_diff = Mock(b_path='foo/features/contrib/user_bob/feature_1.py')
        self.assertTrue(checker.do_check(mock_diff))

        bad_paths = [
            'foo/features/contrib/user_bob/foo1.py',
            'foo/features/contrib/user_bob/1.py',
            'foo/features/contrib/user_bob/feature_x-1.py',
        ]
        for path in bad_paths:
            mock_diff = Mock(b_path=path)
            self.assertFalse(checker.do_check(mock_diff))

    def test_relative_name_depth_check(self):
        checker = RelativeNameDepthCheck(self.project)

        mock_diff = Mock(b_path='foo/features/contrib/user_bob/feature_1.py')
        self.assertTrue(checker.do_check(mock_diff))

        mock_diff = Mock(
            b_path='foo/features/contrib/user_bob/a/b/c/d/feature_1.py')
        self.assertFalse(checker.do_check(mock_diff))
