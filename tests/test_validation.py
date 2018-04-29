import unittest

from fhub_core.validation import PullRequestFeatureValidator

class TestPullRequestFeatureValidator(unittest.TestCase):

    def setUp(self):
        pass

    @unittest.expectedFailure
    def test_todo(self):
        raise NotImplementedError
