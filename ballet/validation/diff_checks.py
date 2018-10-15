import importlib.machinery
from abc import ABCMeta, abstractmethod

from funcy import ignore, re_test

from ballet.compat import pathlib
from ballet.project import relative_to_contrib
from ballet.util.fs import isemptyfile

FEATURE_MODULE_NAME_REGEX = r'feature_[a-zA-Z0-9_]+\.\w+'
SUBPACKAGE_NAME_REGEX = r'user_[a-zA-Z0-9_]+'


class DiffCheck(metaclass=ABCMeta):

    def __init__(self, project):
        self.project = project

    @ignore(Exception, default=False)
    def do_check(self, diff):
        return self.check(diff)

    @abstractmethod
    def check(self, diff):
        pass


class IsAdditionCheck(DiffCheck):

    def check(self, diff):
        return diff.change_type == 'A'


class IsPythonSourceCheck(DiffCheck):

    def check(self, diff):
        path = diff.b_path
        return any(
            path.endswith(ext)
            for ext in importlib.machinery.SOURCE_SUFFIXES
        )


class WithinContribCheck(DiffCheck):

    def check(self, diff):
        path = diff.b_path
        contrib_path = self.project.contrib_module_path
        return pathlib.Path(contrib_path) in pathlib.Path(path).parents


class SubpackageNameCheck(DiffCheck):

    def check(self, diff):
        relative_path = relative_to_contrib(diff, self.project)
        subpackage_name = relative_path.parts[0]
        return re_test(SUBPACKAGE_NAME_REGEX, subpackage_name)


class RelativeNameDepthCheck(DiffCheck):

    def check(self, diff):
        relative_path = relative_to_contrib(diff, self.project)
        return len(relative_path.parts) == 2


class ModuleNameCheck(DiffCheck):

    def check(self, diff):
        filename = pathlib.Path(diff.b_path).parts[-1]
        is_valid_feature_module_name = re_test(
            FEATURE_MODULE_NAME_REGEX, filename)
        is_valid_init_module_name = filename == '__init__.py'
        return is_valid_feature_module_name or is_valid_init_module_name


class IfInitModuleThenIsEmptyCheck(DiffCheck):

    def check(self, diff):
        path = pathlib.Path(diff.b_path)
        filename = path.parts[-1]
        if filename == '__init__.py':
            abspath = self.project.path.joinpath(path)
            return isemptyfile(abspath)
        else:
            return True
