import importlib.machinery
import pathlib

from funcy import re_test

from ballet.project import relative_to_contrib
from ballet.util.fs import isemptyfile
from ballet.validation.base import BaseCheck

FEATURE_MODULE_NAME_REGEX = r'feature_(?P<featurename>[a-zA-Z0-9_]+)\.py'
SUBPACKAGE_NAME_REGEX = r'user_(?P<username>[a-zA-Z0-9_]+)'


class ProjectStructureCheck(BaseCheck):
    """Base class for implementing new Feature API checks

    Args:
        project (Project): project
    """

    def __init__(self, project):
        self.project = project


class IsAdditionCheck(ProjectStructureCheck):

    def check(self, diff):
        """Check that the diff represents the addition of a new file"""
        assert diff.change_type == 'A'


class IsPythonSourceCheck(ProjectStructureCheck):

    def check(self, diff):
        """Check that the new file introduced is a python source file"""
        path = diff.b_path
        assert any(
            path.endswith(ext)
            for ext in importlib.machinery.SOURCE_SUFFIXES
        )


class WithinContribCheck(ProjectStructureCheck):

    def check(self, diff):
        """Check that the new file is within the contrib subdirectory"""
        path = diff.b_path
        contrib_path = self.project.config.get('contrib.module_path')
        assert pathlib.Path(contrib_path) in pathlib.Path(path).parents


class SubpackageNameCheck(ProjectStructureCheck):

    def check(self, diff):
        """Check that the name of the subpackage within contrib is valid

        The package name must match ``user_[a-zA-Z0-9_]+``.
        """
        relative_path = relative_to_contrib(diff, self.project)
        subpackage_name = relative_path.parts[0]
        assert re_test(SUBPACKAGE_NAME_REGEX, subpackage_name)


class RelativeNameDepthCheck(ProjectStructureCheck):

    def check(self, diff):
        """Check that the new file introduced is at the proper depth

        The proper depth is 2 (contrib/user_example/new_file.py)
        """
        relative_path = relative_to_contrib(diff, self.project)
        assert len(relative_path.parts) == 2


class ModuleNameCheck(ProjectStructureCheck):

    def check(self, diff):
        r"""Check that the new file introduced has a valid name

        The module can either be an __init__.py file or must
        match ``feature_[a-zA-Z0-9_]+\.\w+``.
        """
        filename = pathlib.Path(diff.b_path).parts[-1]
        is_valid_feature_module_name = re_test(
            FEATURE_MODULE_NAME_REGEX, filename)
        is_valid_init_module_name = filename == '__init__.py'
        assert is_valid_feature_module_name or is_valid_init_module_name


class IfInitModuleThenIsEmptyCheck(ProjectStructureCheck):

    def check(self, diff):
        """Check that if the new file is __init__.py, then it is empty"""
        path = pathlib.Path(diff.b_path)
        filename = path.parts[-1]
        if filename == '__init__.py':
            abspath = self.project.path.joinpath(path)
            assert isemptyfile(abspath)
