import unittest

import numpy as np

from ballet.validation.alpha_investing import (
    compute_ai, compute_parameters)


class AlphaInvestingTest(unittest.TestCase):

    def test_compute_ai_r(self):
        outcomes = ['rejected']
        w1 = w0 = da = 0.5
        a1 = w1/2
        w2 = w1 - a1
        a2 = w2/(2*2)
        actual = compute_ai(outcomes, w0, da)
        self.assertAlmostEqual(actual, a2, places=3)

    def test_compute_ai_a(self):
        outcomes = ['accepted']
        w1 = w0 = da = 0.5
        a1 = w1/2
        w2 = w1 - a1 + da
        a2 = w2/(2*2)
        actual = compute_ai(outcomes, w0, da)
        self.assertAlmostEqual(actual, a2, places=3)

    def test_compute_parameters_r10(self):
        outcomes = ['rejected'] * 10
        w0 = da = 0.5
        ais, wis = compute_parameters(outcomes, w0=w0, da=da)
        # wealth should be strictly decreasing
        self.assertTrue(
            np.all(np.diff(np.array(wis)) < 0)
        )
