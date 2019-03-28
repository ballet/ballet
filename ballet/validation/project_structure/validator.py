from ballet.validation.base import BaseValidator
from ballet.validation.common import ChangeCollector


class ProjectStructureValidator(BaseValidator):

    def __init__(self, project):
        self.change_collector = ChangeCollector(project)

    def validate(self):
        changes = self.change_collector.collect_changes()
        return not changes.inadmissible_diffs
