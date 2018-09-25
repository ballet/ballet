import unittest

import numpy as np
import pandas as pd

import ballet.eng.ts
from ballet.util.testing import ArrayLikeEqualityTestingMixin


class TestTimeSeriesTransformers(
        unittest.TestCase, ArrayLikeEqualityTestingMixin):

    def test_single_lagger(self):
        # simple test
        data = pd.util.testing.makeTimeSeries()

        trans = ballet.eng.ts.SingleLagger(1)
        result = trans.fit_transform(data)
        expected_result = data.shift(1)

        self.assertSeriesEqual(result, expected_result)

        data = pd.DataFrame(
            data={
                'city': ['LA', 'LA', 'LA', 'NYC', 'BOS', 'BOS', 'BOS'],
                'year': [2001, 2002, 2003, 2002, 2003, 2004, 2005],
                'width': [1, 2, 3, 4, 5, 6, 7],
            }
        ).set_index(['city', 'year']).sort_index()
        trans = ballet.eng.ts.SingleLagger(
            1, groupby_kwargs={'level': 'city'})
        result = trans.fit_transform(data)
        expected_result = pd.DataFrame(
            data={
                'city': ['LA', 'LA', 'LA', 'NYC', 'BOS', 'BOS', 'BOS'],
                'year': [2001, 2002, 2003, 2002, 2003, 2004, 2005],
                'width': [np.nan, 1, 2, np.nan, np.nan, 5, 6],
            }
        ).set_index(['city', 'year']).sort_index()

        self.assertFrameEqual(result, expected_result)

    @unittest.expectedFailure
    def test_multi_lagger(self):
        # TODO
        raise AssertionError
