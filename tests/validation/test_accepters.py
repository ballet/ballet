import unittest
from unittest.mock import patch

from ballet.eng.misc import IdentityTransformer
from ballet.feature import Feature
from ballet.validation.feature_acceptance.validator import (
    GFSSFAccepter, NeverAccepter, RandomAccepter)
from tests.util import load_regression_data


class NoOpAccepterTest(unittest.TestCase):

    def test_accepter(self):
        X = None
        y = None
        existing_features = []
        feature = None

        expected = False

        accepter = NeverAccepter(X, y, existing_features, feature)
        actual = accepter.judge()

        self.assertEqual(expected, actual)


class RandomAccepterTest(unittest.TestCase):

    @patch('random.uniform')
    def test_accepter(self, mock_uniform):
        X = None
        y = None
        existing_features = []
        candidate_feature = None

        expected = True

        mock_uniform.return_value = 0.0  # makes sure feature is accepted

        accepter = RandomAccepter(X, y, existing_features, candidate_feature)
        actual = accepter.judge()

        self.assertEqual(expected, actual)


class GFSSFAccepterTest(unittest.TestCase):

    def setUp(self):
        self.X, self.y = load_regression_data(n_informative=1,
                                              n_uninformative=14)

    def test_init(self):
        feature_1 = Feature(
            input='A_0',
            transformer=IdentityTransformer(),
            source='1st Feature')
        feature_2 = Feature(
            input='Z_0',
            transformer=IdentityTransformer(),
            source='2nd Feature')

        features = [feature_1]
        candidate_feature = feature_2

        accepter = GFSSFAccepter(self.X, self.y, features, candidate_feature)

        self.assertIsNotNone(accepter)
