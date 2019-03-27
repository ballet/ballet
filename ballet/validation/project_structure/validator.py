from ballet.validation.base import BaseValidator
from ballet.validation.common import ChangeCollector


class FileChangeValidator(BaseValidator):

    def __init__(self, project):
        self.change_collector = ChangeCollector(project)

    def validate(self):
        collected_changes = self.change_collector.collect_changes()
        return not collected_changes.inadmissible_diffs
